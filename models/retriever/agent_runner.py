import time
import re
from typing import Dict, List, Optional, Callable

from utils.logger import logger

def _extract_char_keywords(text: str) -> List[str]:
    """Extract meaningful substrings for Chinese/mixed text matching.
    
    Instead of splitting by spaces (which doesn't work for Chinese),
    we use the full text and individual characters/words as matching units.
    For Chinese: we use 2-4 character n-grams as keywords.
    For Latin text: normal word splitting.
    """
    import re
    keywords = []
    # Extract CJK character sequences (2-4 chars)
    cjk_seqs = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]{2,4}', text)
    keywords.extend(cjk_seqs)
    # Extract Latin words (>1 char)
    latin_words = re.findall(r'[a-zA-Z0-9]{2,}', text.lower())
    keywords.extend(latin_words)
    return list(set(keywords))


def cluster_chunks_by_paper(chunks: List[str]) -> tuple:
    """Cluster chunks by their paper title and assign an index.
    
    Only groups chunks by title — metadata enrichment (authors, source, year)
    is done downstream (in backend.py) using the chunk audit file,
    so that the embedding text remains clean and free of structured metadata noise.

    Returns:
        tuple: (paper_list: List[dict], context_str: str)
    """
    import re
    paper_map = {}
    ordered_papers = []
    
    for chunk in chunks:
        # Extract title from chunk string (Title + Abstract format)
        title_match = re.search(r'Title:\s*(.*?)(?:\n|\\n|Abstract:|$)', chunk, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else "Unknown Paper"
        
        if title not in paper_map:
            paper_map[title] = []
            ordered_papers.append(title)
        
        if chunk not in paper_map[title]:
            paper_map[title].append(chunk)
    
    paper_list = []
    context_str = ""
    for i, title in enumerate(ordered_papers):
        paper_index = i + 1
        paper_info = {
            "index":   paper_index,
            "title":   title,
            "authors": "",   # enriched later via chunk audit file in backend
            "source":  "",
            "year":    "",
            "chunks":  paper_map[title]
        }
        paper_list.append(paper_info)
        
        context_str += f"\n论文 [{paper_index}]: {title}\n"
        for chunk_content in paper_map[title]:
            context_str += f"内容: {chunk_content}\n"
            
    return paper_list, context_str


def rerank_chunks_by_keywords(chunks: List[str], question: str, top_k: int) -> List[str]:
    """Rerank chunks by keyword matching with the question.
    
    Uses character-level n-gram matching to support Chinese text,
    since Chinese text cannot be split by spaces.
    """
    if len(chunks) <= top_k:
        return chunks
    
    question_keywords = _extract_char_keywords(question)
    if not question_keywords:
        return chunks[:top_k]
    
    scored_chunks = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = sum(1 for keyword in question_keywords if keyword in chunk_lower)
        scored_chunks.append((chunk, score))
    
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [scored_chunk[0] for scored_chunk in scored_chunks[:top_k]]


def hybrid_rerank_triples(triples: List[str], question: str, top_k: int) -> List[str]:
    """Rerank triples using both keyword matching and embedded vector scores.
    
    Each formatted triple string may contain a '[score: X.XXX]' suffix from
    the retriever's cosine similarity. We parse that score and combine it
    with a keyword-match score for a more accurate ranking.
    
    Args:
        triples: List of formatted triple strings, optionally containing '[score: X]'
        question: Original user question
        top_k: Maximum number of triples to return
    """
    import re
    if len(triples) <= top_k:
        return triples
    
    q_keywords = _extract_char_keywords(question)
    
    scored = []
    for triple in triples:
        # Extract vector score embedded in triple string if present
        vec_score = 0.0
        score_match = re.search(r'\[score:\s*([0-9.]+)\]', triple)
        if score_match:
            try:
                vec_score = float(score_match.group(1))
            except ValueError:
                pass
        
        # Keyword match score (normalised to ~[0, 1])
        kw_hits = sum(1 for kw in q_keywords if kw in triple.lower()) if q_keywords else 0
        kw_score = kw_hits / max(len(q_keywords), 1)
        
        # Combined: 60% vector similarity + 40% keyword match
        combined = 0.6 * vec_score + 0.4 * kw_score
        scored.append((triple, combined))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in scored[:top_k]]

def deduplicate_triples(triples: List[str]) -> List[str]:
    return list(dict.fromkeys(triples))

def merge_chunk_contents(chunk_ids: List[str], chunk_contents_dict: Dict[str, str]) -> List[str]:
    return [chunk_contents_dict.get(str(chunk_id), f"[Missing content for chunk {chunk_id}]") for chunk_id in chunk_ids]

def initial_question_decomposition(graphq, kt_retriever, question: str, schema_path: str, top_k: int, callback: Callable = None) -> dict:
    """
    Process a single question using noagent mode and return structured results.
    """
    all_triples = set()
    all_chunk_ids = set()
    all_chunk_contents = dict()
    all_sub_question_results = []
    total_time = 0

    try:
        decomposition_result = graphq.decompose(question, schema_path)
        sub_questions = decomposition_result.get("sub_questions", [])
        involved_types = decomposition_result.get("involved_types", {})
        decompose_fallback = False
        decompose_error = None
        if callback:
            callback("decompose", sub_questions_count=len(sub_questions), sub_questions=[sq.get("sub-question", "") for sq in sub_questions][:5], decompose_fallback=False, schema_path=schema_path)
        logger.info(f"Original question: {question}")
        logger.info(f"Decomposed into {len(sub_questions)} sub-questions")
    except Exception as e:
        logger.error(f"Error decomposing question: {str(e)}")
        sub_questions = [{"sub-question": question}]
        involved_types = {"nodes": [], "relations": [], "attributes": []}  
        decomposition_result = {"sub_questions": sub_questions, "involved_types": involved_types}
        decompose_fallback = True
        decompose_error = str(e)
        if callback:
            callback("decompose", decompose_fallback=True, decompose_error=decompose_error[:300], schema_path=schema_path)

    if len(sub_questions) > 1:
        logger.info("Using parallel sub-question processing...")
        if callback:
            callback("retrieval_progress", progress=65, message="Parallel initial retrieval...")
        aggregated_results, parallel_time = kt_retriever.process_subquestions_parallel(
            sub_questions, top_k=top_k, involved_types=involved_types
        )
        total_time += parallel_time
        all_triples.update(aggregated_results['triples'])
        all_chunk_ids.update(aggregated_results['chunk_ids'])
        for chunk_id, content in aggregated_results['chunk_contents'].items():
            all_chunk_contents[str(chunk_id)] = content
        all_sub_question_results = aggregated_results['sub_question_results']
        
        # We need to adapt the parallel results to the structure expected by the frontend
        for idx, sq_res in enumerate(all_sub_question_results):
            sq_res["type"] = "sub_question"
            sq_res["index"] = idx + 1
            sq_res["processing_time"] = sq_res.get("time_taken", 0)
            sq_res["chunks_count"] = sq_res.get("chunk_ids_count", 0)
            
        if callback:
            # We don't have per-subquestion updates here easily, so we just send one for parallel
            pass

    else:
        logger.info("Using single sub-question processing...")
        if callback:
            callback("retrieval_progress", progress=65, message="Initial retrieval...")
            
        for i, sub_question in enumerate(sub_questions):
            try:
                sub_question_text = sub_question.get("sub-question", question)
                logger.info(f"Processing sub-question {i+1}: {sub_question_text}")
                retrieval_results, time_taken = kt_retriever.process_retrieval_results(sub_question_text, top_k=top_k, involved_types=involved_types)
                total_time += time_taken
                triples = retrieval_results.get('triples', []) or []
                chunk_ids = retrieval_results.get('chunk_ids', []) or []
                chunk_contents = retrieval_results.get('chunk_contents', []) or []
                diagnostics = retrieval_results.get("diagnostics", {}) or {}
                
                step_chunk_contents = []
                if isinstance(chunk_contents, dict):
                    for chunk_id, content in chunk_contents.items():
                        all_chunk_contents[str(chunk_id)] = content
                        step_chunk_contents.append(content)
                else:
                    for i_c, chunk_id in enumerate(chunk_ids):
                        if i_c < len(chunk_contents):
                            all_chunk_contents[str(chunk_id)] = chunk_contents[i_c]
                            step_chunk_contents.append(chunk_contents[i_c])

                sub_result = {
                    'type': 'sub_question',
                    'index': i + 1,
                    'question': sub_question_text,
                    'sub_question': sub_question_text,
                    'triples': triples[:10],
                    'triples_count': len(triples),
                    'chunk_ids': [str(c) for c in chunk_ids],
                    'chunk_contents': step_chunk_contents,
                    'chunks_count': len(chunk_ids),
                    'processing_time': time_taken,
                    'retrieval_diagnostics': diagnostics,
                    'time_taken': time_taken
                }
                all_sub_question_results.append(sub_result)
                all_triples.update(triples)
                all_chunk_ids.update(str(c) for c in chunk_ids)
                
                if callback:
                    callback("sub_question_update", 
                             index=i+1, 
                             total=len(sub_questions), 
                             question=sub_question_text, 
                             triples_preview=list(dict.fromkeys(triples))[:5],
                             triples_count=len(triples),
                             chunks_count=len(chunk_ids),
                             retrieval_diagnostics=diagnostics,
                             processing_time=time_taken)

            except Exception as e:
                logger.error(f"Error processing sub-question {i+1}: {str(e)}")
                sub_question_text = sub_question.get('sub-question', question)
                sub_result = {
                    'type': 'sub_question',
                    'index': i + 1,
                    'question': sub_question_text,
                    'sub_question': sub_question_text,
                    'triples_count': 0,
                    'chunks_count': 0,
                    'chunk_ids_count': 0,
                    'processing_time': 0.0,
                    'time_taken': 0.0,
                    'error': str(e)
                }
                all_sub_question_results.append(sub_result)
                continue
            
    dedup_triples = deduplicate_triples(list(all_triples))
    dedup_chunk_ids = list(dict.fromkeys(all_chunk_ids))
    dedup_chunk_contents = merge_chunk_contents(dedup_chunk_ids, all_chunk_contents)

    if not dedup_triples and not dedup_chunk_contents:
        logger.warning(f"No triples or chunks retrieved for question: {question}")
        dedup_triples = ["No relevant information found"]
        dedup_chunk_contents = ["No relevant chunks found"]

    if len(dedup_triples) > top_k:
        dedup_triples = hybrid_rerank_triples(dedup_triples, question, top_k)

    if len(dedup_chunk_contents) > top_k:
        dedup_chunk_contents = rerank_chunks_by_keywords(dedup_chunk_contents, question, top_k)

    paper_list, chunks_context = cluster_chunks_by_paper(dedup_chunk_contents)
    context = "=== Triples ===\n" + "\n".join(dedup_triples)
    context += "\n=== Chunks ===\n" + chunks_context

    prompt = kt_retriever.generate_prompt(question, context)

    max_retries = 3
    answer = None
    for retry in range(max_retries):
        try:
            answer = kt_retriever.generate_answer(prompt)
            if answer and answer.strip():
                break
        except Exception as e:
            logger.error(f"Error generating answer (attempt {retry + 1}): {str(e)}")
            if retry == max_retries - 1:
                answer = f"Error: Unable to generate answer - {str(e)}"
            time.sleep(1)

    return {
        'decomposition_result': decomposition_result,
        'sub_questions': sub_questions,
        'involved_types': involved_types,
        'triples': dedup_triples,
        'chunk_ids': dedup_chunk_ids,
        'chunk_contents': dedup_chunk_contents,
        'sub_question_results': all_sub_question_results,
        'answer': answer,
        'total_time': total_time,
        'decompose_fallback': decompose_fallback,
        'decompose_error': decompose_error,
        'all_chunk_contents_dict': all_chunk_contents,
        'all_chunk_ids_set': all_chunk_ids,
        'all_triples_set': all_triples,
        'retrieved_papers': paper_list
    }

def run_agent_retrieval(graphq, kt_retriever, question: str, schema_path: str, max_steps: int, top_k: int, callback: Callable = None) -> dict:
    """
    Run the full IRCoT agent retrieval logic.
    """
    total_time = 0
    thoughts = []
    logs = []
    
    logger.info(f"Starting Agent mode for question: {question}")
    
    # Step 0: Initial analysis
    initial_result = initial_question_decomposition(graphq, kt_retriever, question, schema_path, top_k, callback)
    total_time += initial_result['total_time']
    
    all_triples = initial_result['all_triples_set']
    all_chunk_ids = initial_result['all_chunk_ids_set']
    all_chunk_contents = initial_result['all_chunk_contents_dict']
    
    # Variables to hold the increment for the current step
    current_new_triples = deduplicate_triples(list(all_triples))
    current_new_chunk_ids = list(dict.fromkeys(all_chunk_ids))
    current_new_chunk_contents = merge_chunk_contents(current_new_chunk_ids, all_chunk_contents)
    
    reasoning_steps = list(initial_result['sub_question_results'])
    
    # Use full initial answer without truncation in thoughts
    initial_thought = f"Initial analysis: {initial_result['answer']}"
    thoughts.append(initial_thought)
    
    if callback:
        callback("retrieval_progress", progress=75, message="Iterative reasoning...")
        callback("ircot_start", message="Starting iterative reasoning")
        
    current_query = question
    final_answer = initial_result['answer']
    
    for step in range(1, max_steps + 1):
        logger.info(f"IRCoT Step {step}/{max_steps}")
        
        # Build context using ONLY the increment from the previous retrieval step
        paper_list, chunks_context = cluster_chunks_by_paper(current_new_chunk_contents[:10])
        context = "=== Triples ===\n" + "\n".join(current_new_triples[:20]) # Limit context length to avoid token explosion
        context += "\n=== Chunks ===\n" + chunks_context
        
        ircot_prompt = f"""
You are an expert knowledge assistant using iterative retrieval with chain-of-thought reasoning.

Current Question: {question}
Current Iteration Query: {current_query}

Available Knowledge Context:
{context}

Previous Thoughts: {' | '.join(thoughts) if thoughts else 'None'}

Step {step}: Please think step by step about what additional information you need to answer the question completely and accurately.

Instructions:
1. Analyze the current knowledge context and the question
2. Consider the previous thoughts and analysis
3. Think about what information might be missing or unclear
4. If you have enough information to answer, in the end of your response, write "So the answer is:" followed by your final answer
5. If you need more information, in the end of your response, write a specific query begin with "The new query is:" to retrieve additional relevant information
6. Be specific and focused in your reasoning

Your reasoning:
"""
        max_retries = 3
        reasoning = None
        for retry in range(max_retries):
            try:
                reasoning = kt_retriever.generate_answer(ircot_prompt)
                if reasoning and reasoning.strip():
                    break
            except Exception as e:
                logger.error(f"Error generating IRCoT response (attempt {retry + 1}): {str(e)}")
                if retry == max_retries - 1:
                    reasoning = f"Error: Unable to generate reasoning - {str(e)}"
                time.sleep(1)
        
        # Append full reasoning, NO TRUNCATION
        thoughts.append(reasoning)
        
        reasoning_steps.append({
            "type": "ircot_step",
            "step": step,
            "question": current_query,
            "triples": current_new_triples[:10],
            "triples_count": len(current_new_triples),
            "chunks_count": len(current_new_chunk_ids),
            "processing_time": 0,
            "chunk_contents": current_new_chunk_contents[:3],
            "thought": reasoning[:500] # Slightly truncated for frontend display only, not the actual LLM chain
        })
        
        if callback:
            callback("ircot_update", step=step, max_steps=max_steps, current_query=current_query, thought_preview=(reasoning or "")[:500])
        
        if "So the answer is:" in reasoning:
            match = re.search(r"So the answer is:\s*(.*)", reasoning, flags=re.IGNORECASE | re.DOTALL)
            final_answer = match.group(1).strip() if match else reasoning
            logger.info("Final answer found, stopping IRCoT.")
            break

        if "The new query is:" in reasoning:
            lines = reasoning.split("The new query is:")[1].strip().splitlines()
            new_query = lines[0] if lines else ""
        else:
            new_query = reasoning
        
        if new_query and new_query != current_query:
            current_query = new_query
            logger.info(f"New query for next iteration: {current_query}")
            
            if callback:
                callback("retrieval_progress", progress=min(90, 75 + step * 5), message=f"Iterative retrieval step {step}...")
                
            retrieval_results, time_taken = kt_retriever.process_retrieval_results(current_query, top_k=top_k)
            total_time += time_taken
            
            new_triples = retrieval_results.get('triples', []) or []
            new_chunk_ids = retrieval_results.get('chunk_ids', []) or []
            new_chunk_contents = retrieval_results.get('chunk_contents', []) or []
            
            if isinstance(new_chunk_contents, list):
                new_chunk_contents_dict = {}
                new_chunk_contents_list = []
                for i, chunk_id in enumerate(new_chunk_ids):
                    if i < len(new_chunk_contents):
                        content = new_chunk_contents[i]
                    else:
                        content = f"[Missing content for chunk {chunk_id}]"
                    new_chunk_contents_dict[str(chunk_id)] = content
                    new_chunk_contents_list.append(content)
            else:
                new_chunk_contents_dict = {str(k): v for k, v in new_chunk_contents.items()}
                new_chunk_contents_list = [new_chunk_contents_dict.get(str(cid), "") for cid in new_chunk_ids]
            
            # Extract increments for the next iteration step
            current_new_triples = deduplicate_triples(new_triples)
            current_new_chunk_ids = list(dict.fromkeys([str(cid) for cid in new_chunk_ids]))
            current_new_chunk_contents = new_chunk_contents_list
            
            all_triples.update(new_triples)
            all_chunk_ids.update(str(cid) for cid in new_chunk_ids)
            all_chunk_contents.update(new_chunk_contents_dict)
        else:
            logger.info("No new query generated, stopping IRCoT.")
            break
    
    # Synthesis Step
    # Perform a global rerank on the accumulated knowledge
    final_triples = deduplicate_triples(list(all_triples))
    if len(final_triples) > 30:
        final_triples = hybrid_rerank_triples(final_triples, question, 30)
    else:
        # Just to ensure they are properly ordered by the initial question if less than 30
        final_triples = hybrid_rerank_triples(final_triples, question, len(final_triples))

    final_chunk_ids = list(dict.fromkeys(all_chunk_ids))
    final_chunk_contents = merge_chunk_contents(final_chunk_ids, all_chunk_contents)
    if len(final_chunk_contents) > 15:
        final_chunk_contents = rerank_chunks_by_keywords(final_chunk_contents, question, 15)
    else:
        final_chunk_contents = rerank_chunks_by_keywords(final_chunk_contents, question, len(final_chunk_contents))

    # To ensure consistent slicing for final chunks
    final_triples = final_triples[:30]
    final_chunk_contents = final_chunk_contents[:15]

    if callback:
        callback("retrieval_progress", progress=95, message="Synthesizing final answer...")

    paper_list, chunks_context = cluster_chunks_by_paper(final_chunk_contents)
    final_context = "=== Final Triples ===\n" + "\n".join(final_triples)
    final_context += "\n=== Final Chunks ===\n" + chunks_context
    
    final_prompt = kt_retriever.generate_prompt(question, final_context)
    
    answer = None
    for retry in range(3):
        try:
            answer = kt_retriever.generate_answer(final_prompt)
            if answer and answer.strip():
                break
        except Exception as e:
            logger.error(f"Error generating final synthesis answer: {str(e)}")
            time.sleep(1)
            
    if not answer or not answer.strip():
        answer = final_answer # Fallback to the loop's answer
    
    if callback:
        callback("retrieval_progress", progress=100, message="Answer generation completed!")
        
    return {
        'answer': answer,
        'sub_questions': initial_result.get('sub_questions', []),
        'retrieved_triples': final_triples,
        'retrieved_chunks': final_chunk_contents,
        'reasoning_steps': reasoning_steps,
        'decompose_fallback': initial_result.get('decompose_fallback', False),
        'decompose_error': initial_result.get('decompose_error'),
        'total_time': total_time,
        'thoughts': thoughts,
        'retrieved_papers': paper_list
    }

