export const default_qa_template = `Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:`

export const qa_template = `Use the following pieces of context to answer the question at the end.

{context}

Question: {question}
Helpful Answer:`

export const default_map_reduce_template = `Given the following extracted parts of a long document and a question, create a final answer. 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.

{summaries}

Question: {question}
Helpful Answer:`

export const map_reduce_template = `Given the following extracted parts of a long document and a question, create a final answer. 

{summaries}

Question: {question}
Helpful Answer:`

export const CUSTOM_QUESTION_GENERATOR_CHAIN_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language. include it in the standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`

export const default_convo_chain_prompt = `Given the following conversation and a follow up question, answer the question at the end. If the AI does not know the answer to a question, it truthfully says it does not know.

Chat History:
{chat_history}

Question: {question}
Helpful Answer:`

export const convo_chain_prompt = `Given the following conversation and a follow up question, answer the question at the end.

Chat History:
{chat_history}

Question: {question}
Helpful Answer:`
