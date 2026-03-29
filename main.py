"""
- noagent: Basic retrieval and answer generation
- agent: Question decomposition with parallel sub-question processing and Iterative Retrieval Chain of Thought with step-by-step reasoning
"""
import json
import json_repair
import time
import argparse
import os
import glob
import shutil
from typing import List

from models.constructor import kt_gen as constructor
from models.retriever import agentic_decomposer as decomposer, enhanced_kt_retriever as retriever, agent_runner
from utils.eval import Eval
from config import get_config, ConfigManager
from utils.logger import logger
from utils.process_control import install_interrupt_guard, terminate_process_tree


def tuples_to_string(rows, sep=", ", line_sep="\n", wrap_brackets=True):
    def fmt(t):
        inner = sep.join(map(str, t))
        return f"[{inner}]" if wrap_brackets else inner
    return line_sep.join(fmt(t) for t in rows)


def rerank_chunks_by_keywords(chunks: List[str], question: str, top_k: int) -> List[str]:
    """
    Rerank chunks by keyword matching with the question
    
    Args:
        chunks: List of chunk contents
        question: Original question
        top_k: Number of top chunks to return
        
    Returns:
        Reranked list of chunks
    """
    if len(chunks) <= top_k:
        return chunks
    
    question_keywords = set(question.lower().split())
    scored_chunks = []
    
    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = sum(1 for keyword in question_keywords if keyword in chunk_lower)
        scored_chunks.append((chunk, score))
    
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    return [scored_chunk[0] for scored_chunk in scored_chunks[:top_k]]


def deduplicate_triples(triples: List[str]) -> List[str]:

    return list(set(triples))


def merge_chunk_contents(chunk_ids, chunk_contents_dict):

    return [chunk_contents_dict.get(chunk_id, f"[Missing content for chunk {chunk_id}]") for chunk_id in chunk_ids]


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Youtu-GraphRAG Framework")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/base_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--datasets", 
        nargs="+", 
        default=["demo"],
        help="List of datasets to process"
    )

    parser.add_argument(
        "--override",
        type=str,
        help="JSON string with configuration overrides"
    )
    return parser.parse_args()


def setup_environment(config: ConfigManager):
    """Set up the environment based on configuration."""
    config.create_output_directories()
    
    logger.info("Youtu-GraphRAG initialized")
    logger.info(f"Mode: {config.triggers.mode}")
    logger.info(f"Constructor enabled: {config.triggers.constructor_trigger}")
    logger.info(f"Retriever enabled: {config.triggers.retrieve_trigger}")


def clear_cache_files(dataset_name: str) -> None:
    """Clear cache files for a dataset before graph construction (CLI path)."""
    try:
        faiss_cache_dir = f"retriever/faiss_cache_new/{dataset_name}"
        if os.path.exists(faiss_cache_dir):
            shutil.rmtree(faiss_cache_dir)
            logger.info(f"Cleared FAISS cache directory: {faiss_cache_dir}")

        chunk_file = f"output/chunks/{dataset_name}.txt"
        if os.path.exists(chunk_file):
            os.remove(chunk_file)
            logger.info(f"Cleared chunk file: {chunk_file}")

        graph_file = f"output/graphs/{dataset_name}_new.json"
        if os.path.exists(graph_file):
            os.remove(graph_file)
            logger.info(f"Cleared graph file: {graph_file}")

        cache_patterns = [
            f"output/logs/{dataset_name}_*.log",
            f"output/chunks/{dataset_name}_*",
            f"output/graphs/{dataset_name}_*",
        ]
        for pattern in cache_patterns:
            for file_path in glob.glob(pattern):
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logger.info(f"Cleared cache file: {file_path}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        logger.info(f"Cleared cache directory: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clear {file_path}: {e}")

        logger.info(f"Cache cleanup completed for dataset: {dataset_name}")

    except Exception as e:
        logger.error(f"Error clearing cache files for {dataset_name}: {e}")


def graph_construction(datasets):
    if config.triggers.constructor_trigger:
        logger.info("Starting knowledge graph construction...")
        
        for dataset in datasets:
            
            try:
                dataset_config = config.get_dataset_config(dataset)
                logger.info(f"Building knowledge graph for dataset: {dataset}")
                logger.info("Clearing caches before construction...")
                clear_cache_files(dataset)
                
                builder = constructor.KTBuilder(
                    dataset, 
                    dataset_config.schema_path, 
                    mode=config.construction.mode,
                    config=config
                )

                builder.build_knowledge_graph(dataset_config.corpus_path)
                logger.info(f"Successfully built knowledge graph for {dataset}")
            
            except Exception as e:
                logger.error(f"Failed to build knowledge graph for {dataset}: {e}")
                continue
    return


def retrieval(datasets):
    for dataset in datasets:
        dataset_config = config.get_dataset_config(dataset)
        
        with open(dataset_config.qa_path, "r", encoding="utf-8") as f:
            qa_pairs = json_repair.load(f)
        
        # evaluator = Eval(config.api.llm_api_key)
        graphq = decomposer.GraphQ(dataset, config=config)
        
        logger.info("🚀 Initializing retriever 🚀")
        logger.info("-"*30)
        
        kt_retriever = retriever.KTRetriever(
            dataset, 
            dataset_config.graph_output, 
            recall_paths=config.retrieval.recall_paths,
            schema_path=dataset_config.schema_path, 
            top_k=config.retrieval.top_k_filter, 
            mode=config.triggers.mode,
            config=config
        )
        
        logger.info("🚀 Building FAISS index 🚀")
        logger.info("-"*30)
        start_time = time.time()
        kt_retriever.build_indices()
        logger.info(f"Time taken to build FAISS index: {time.time() - start_time} seconds")
        logger.info("-"*30)
        
        logger.info(f"Start answering questions...")
        logger.info("-"*30)
    
        if config.triggers.mode == "noagent":
            no_agent_retrieval(graphq, kt_retriever, qa_pairs, dataset_config.schema_path)

        elif config.triggers.mode == "agent":
            agent_retrieval(graphq, kt_retriever, qa_pairs, dataset_config.schema_path)


def no_agent_retrieval(graphq, kt_retriever, qa_pairs, schema_path):
    total_time = 0
    accuracy = 0
    total_questions = len(qa_pairs)
    evaluator = Eval()
    for qa in qa_pairs:
        result = agent_runner.initial_question_decomposition(graphq, kt_retriever, qa["question"], schema_path, config.retrieval.top_k_filter)
        total_time += result['total_time']

        logger.info(f"========== Original Question: {qa['question']} ==========") 
        logger.info(f"Gold Answer: {qa['answer']}")
        logger.info(f"Generated Answer: {result['initial_answer']}")
        logger.info("-" * 30)

        eval_result = evaluator.eval(qa["question"], qa["answer"], result['initial_answer'])
        logger.info(f"No agent mode eval result: {eval_result}")
        if eval_result == "1":
            accuracy += 1
    logger.info(f"Eval result: {'Correct' if eval_result == '1' else 'Wrong'}")
    logger.info(f"Overall Accuracy: {accuracy / total_questions * 100}%")     
    logger.info(f"Average time taken: {total_time / total_questions} seconds")

def agent_retrieval(graphq, kt_retriever, qa_pairs, schema_path):
    total_time = 0
    accuracy = 0
    total_questions = len(qa_pairs)
    evaluator = Eval()
    max_steps = getattr(config.retrieval.agent, 'max_steps', 3)
                    
    for qa in qa_pairs:
        result = agent_runner.run_agent_retrieval(
            graphq, 
            kt_retriever, 
            qa["question"], 
            schema_path, 
            max_steps, 
            config.retrieval.top_k_filter
        )
        total_time += result['total_time']
        
        logger.info(f"========== Original Question: {qa['question']} ==========") 
        logger.info(f"IRCoT Steps: {len(result['thoughts']) - 1}")
        logger.info(f"Final Triples: {len(result['retrieved_triples'])}")
        logger.info(f"Final Chunks: {len(result['retrieved_chunks'])}")
        logger.info(f"Gold Answer: {qa['answer']}")
        logger.info(f"Generated Answer: {result['answer']}")
        logger.info(f"Thought Process: {' | '.join(result['thoughts'])}")
        logger.info("-" * 30)
        
        eval_result = evaluator.eval(qa["question"], qa["answer"], result['answer'])
        logger.info(f"Agent mode eval result: {eval_result}")
        if eval_result == "1":
            accuracy += 1
    logger.info(f"Eval result: {'Correct' if eval_result == '1' else 'Wrong'}")
    logger.info(f"Overall Accuracy: {accuracy / total_questions * 100}%")
    logger.info(f"Average time taken: {total_time / total_questions} seconds")

if __name__ == "__main__":

    install_interrupt_guard("Youtu-GraphRAG CLI")
    args = parse_arguments()
    config_path = args.config
    try:
        config = get_config(config_path)

        if args.override:
            try:
                overrides = json.loads(args.override)
                config.override_config(overrides)
                logger.info("Applied configuration overrides")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in override parameter: {e}")
                raise SystemExit(1)

        setup_environment(config)

        datasets = args.datasets

        # ########### Construction ###########
        if config.triggers.constructor_trigger:
            logger.info("Starting knowledge graph construction...")
            graph_construction(datasets)

        # ########### Retriever ###########
        if config.triggers.retrieve_trigger:
            logger.info("Starting knowledge retrieval and QA...")
            retrieval(datasets)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Forcing shutdown...")
        terminate_process_tree()
