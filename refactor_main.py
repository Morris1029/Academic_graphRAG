import sys

with open('main.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'from models.retriever import agentic_decomposer as decomposer, enhanced_kt_retriever as retriever' in line:
        new_lines.append(line)
        new_lines.append('from models.retriever import agent_runner\n')
    else:
        new_lines.append(line)

lines = new_lines

start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if 'def initial_question_decomposition(' in line:
        start_idx = i
    if 'def agent_retrieval(' in line:
        end_idx = i - 1
        break

if start_idx != -1 and end_idx != -1:
    # Just replace it entirely
    pass

# We actually want to replace both `no_agent_retrieval`, `initial_question_decomposition`, and `agent_retrieval`?
# In main.py:
# def initial_question_decomposition(...) -> lines ~210 to 351
# def no_agent_retrieval(...) -> lines ~354 to ~376
# def agent_retrieval(...) -> lines ~378 to ~550
# We can just replace agent_retrieval and no_agent_retrieval with calls to the new ones!

# Wait, since main.py uses initial_question_decomposition inside no_agent_retrieval, we can just replace everything with agents.

with open('main.py', 'r', encoding='utf-8') as f:
    text = f.read()

import re

# We will just write a simpler replace logic.

replacement = '''
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
'''

start_marker = "def no_agent_retrieval("
end_marker = "if __name__ == \"__main__\":"

s = text.find(start_marker)
e = text.find(end_marker)

if s != -1 and e != -1:
    new_text = text[:s] + replacement + text[e + len(end_marker):]
    
    # fix imports as well
    if "from models.retriever import agent_runner" not in new_text:
        new_text = new_text.replace(
            "from models.retriever import agentic_decomposer as decomposer, enhanced_kt_retriever as retriever",
            "from models.retriever import agentic_decomposer as decomposer, enhanced_kt_retriever as retriever, agent_runner"
        )
        
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(new_text)
    print("main.py updated!")
else:
    print("markers not found!")

# Also remove def initial_question_decomposition from main.py since it's unused now.
with open('main.py', 'r', encoding='utf-8') as f:
    text = f.read()
    
s2 = text.find("def initial_question_decomposition(")
if s2 != -1:
    e2 = text.find("def no_agent_retrieval(")
    if e2 != -1:
        text = text[:s2] + text[e2:]
        with open('main.py', 'w', encoding='utf-8') as f:
            f.write(text)
        print("deleted initial_question_decomposition from main.py")
