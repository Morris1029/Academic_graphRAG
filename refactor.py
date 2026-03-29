import sys
import datetime

with open('backend.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if 'def _merge_chunk_contents(ids, mapping):' in line and start_idx == -1:
        start_idx = i - 1
    if 'return QuestionResponse(' in line and start_idx != -1:
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    new_lines = lines[:start_idx]
    
    replacement = '''
        # Use unified agent runner
        def _sync_callback(event_type, **kwargs):
            try:
                msg = {"type": "qa_update", "stage": event_type, "timestamp": datetime.now().isoformat(), **kwargs}
                if event_type == "retrieval_progress":
                    progress = kwargs.get("progress", 0)
                    msg_text = kwargs.get("message", "Processing...")
                    asyncio.run_coroutine_threadsafe(
                        send_progress_update(client_id, "retrieval", progress, msg_text),
                        loop
                    )
                else:
                    asyncio.run_coroutine_threadsafe(
                        manager.send_message(msg, client_id),
                        loop
                    )
            except Exception as e:
                pass

        max_steps = getattr(getattr(config.retrieval, 'agent', object()), 'max_steps', 3)
        loop = asyncio.get_running_loop()
        
        # Offload logic to thread executor to avoid blocking event loop
        result = await loop.run_in_executor(
            None, 
            lambda: agent_runner.run_agent_retrieval(
                graphq, 
                kt_retriever, 
                question, 
                schema_path, 
                max_steps, 
                config.retrieval.top_k_filter,
                callback=_sync_callback
            )
        )

        final_answer = result['answer']
        sub_questions = result['sub_questions']
        final_triples = result['retrieved_triples']
        final_chunk_contents = result['retrieved_chunks']
        reasoning_steps = result['reasoning_steps']
        decompose_fallback = result['decompose_fallback']
        decompose_error = result['decompose_error']

        await send_progress_update(client_id, "retrieval", 100, "Answer generation completed!")

        try:
            await manager.send_message({
                "type": "qa_complete",
                "answer_preview": (final_answer or "")[:300],
                "sub_questions_count": len(sub_questions),
                "triples_final_count": len(final_triples),
                "chunks_final_count": len(final_chunk_contents),
                "timestamp": datetime.now().isoformat()
            }, client_id)
        except Exception as _e:
            logger.debug(f"QA complete ws send failed: {_e}")

        visualization_data = {
            "subqueries": prepare_subquery_visualization(sub_questions, reasoning_steps),
            "knowledge_graph": prepare_retrieved_graph_visualization(final_triples),
            "reasoning_flow": prepare_reasoning_flow_visualization(reasoning_steps),
            "retrieval_details": {
                "total_triples": len(final_triples),
                "total_chunks": len(final_chunk_contents),
                "sub_questions_count": len(sub_questions),
                "triples_by_subquery": [s.get("triples_count", 0) for s in reasoning_steps if s.get("type") == "sub_question"],
                "diagnostics_by_subquery": [s.get("retrieval_diagnostics", {}) for s in reasoning_steps if s.get("type") == "sub_question"],
            }
        }

        return QuestionResponse(
'''
    new_lines.append(replacement)
    new_lines.extend(lines[end_idx+1:])
    
    with open('backend.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("Successfully replaced!")
else:
    print(f"Could not find target strings. start={start_idx}, end={end_idx}")
