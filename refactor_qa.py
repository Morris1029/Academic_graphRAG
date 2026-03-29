import sys

with open('eval/rag_eval/qa_runner.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# add import
new_lines = []
for line in lines:
    if 'from models.retriever import enhanced_kt_retriever as retriever' in line:
        new_lines.append(line)
        new_lines.append('from models.retriever import agent_runner\n')
    else:
        new_lines.append(line)

lines = new_lines

# find _run_agent
start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if 'def _run_agent(self, question_id: str, question: str) -> QAPrediction:' in line:
        start_idx = i
    if 'def answer_question(self, question_id: str, question: str) -> QAPrediction:' in line:
        end_idx = i - 1
        break

if start_idx != -1 and end_idx != -1:
    new_lines = lines[:start_idx]
    
    replacement = '''    def _run_agent(self, question_id: str, question: str) -> QAPrediction:
        start_time = time.time()
        max_steps = int(getattr(self.config.retrieval.agent, "max_steps", 3))

        result = agent_runner.run_agent_retrieval(
            self.graphq, 
            self.kt_retriever, 
            question, 
            self.schema_path, 
            max_steps, 
            self.config.retrieval.top_k_filter
        )

        return QAPrediction(
            question_id=question_id,
            answer=result['answer'],
            sub_questions=result['sub_questions'],
            retrieved_triples=result['retrieved_triples'],
            retrieved_chunks=result['retrieved_chunks'],
            reasoning_steps=result['reasoning_steps'],
            decompose_fallback=result['decompose_fallback'],
            decompose_error=result['decompose_error'],
            schema_path_used=self.schema_path,
            latency_seconds=time.time() - start_time,
        )

'''
    new_lines.append(replacement)
    new_lines.extend(lines[end_idx+1:])
    
    with open('eval/rag_eval/qa_runner.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("Successfully replaced in qa_runner.py!")
else:
    print(f"Could not find target strings. start={start_idx}, end={end_idx}")

# Also replace _run_noagent to use agent_runner ?
# _run_noagent currently uses _initial_question_decomposition manually built in qa_runner
# But we can just use agent_runner.initial_question_decomposition directly!
# For minimal impact I'll leave _run_noagent logic as is or replace it if needed. Actually it's better to replace it too to ensure one single truth.
