# OpenSearch Agentic Memory Integration with Strands Agent and LangGraph

These demo scripts showcase OpenSearch's Agentic Memory capabilities with AI agents, providing persistent memory management that enables agents to maintain context across sessions and conversations. The examples demonstrate integration with popular AI frameworks like Strands and LangGraph, showing both short-term and long-term memory patterns.

## Prerequisites

1. Clone the repository

```bash
git clone https://github.com/opensearch-project/opensearch-py-ml.git
cd docs/source/examples/agentic_memory
```

2. Ensure you have Python 3.10+ installed, then

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

## Strands Agents (Short-term memory)

1. Configure environment

```bash
# Copy Strands environment template
cp strands/.env.strands-short.example strands/.env.strands-short

# Edit .env.strands-short file with the required configurations
```

> See strands/.env.strands-short.example for all available configuration options

2. Run the script

```bash
python strands/strands_short_term.py
```

The script creates an interactive chat session where you can converse with the AI agent. The agent maintains conversation history using OpenSearch's persistent memory.

- Enter your messages at the "You:" prompt
- Type `q` or `quit` to exit

**Example conversation:**
```
Created memory container with id 'Wac4o5oB6Os6SYy-CAwu'
OpenSearch Agentic Memory Demo - Session: demo_session
Type 'q' or 'quit' to end the conversation

👤 You: Hello, my name is Bob and I enjoy swimming.
🤖 Assistant: Hi Bob! Nice to meet you. Swimming is a great exercise!

👤 You: Do you remember my name?
🤖 Assistant: Yes, your name is Bob. You mentioned it when you introduced yourself.

👤 You: q
🤖 Assistant: Goodbye!
```

## Strands Agents (Long-term memory)

1. Configure environment

```bash
# Copy Strands environment template
cp strands/.env.strands-long.example strands/.env.strands-long

# Edit .env.strands-long file with the required configurations
```

> See strands/.env.strands-long.example for all available configuration options

2. Run the script

```bash
python strands/strands_long_term.py
```

The script creates an interactive chat session with long-term memory capabilities. Unlike short-term memory, this agent can remember information across different sessions and uses semantic search to retrieve relevant memories based on meaning.

- Enter your messages at the "You:" prompt
- Type `q` or `quit` to exit

**Example conversation:**
```
% python strands/strands_long_term.py
Created embedding model with id 'tqfkp5oB6Os6SYy-CA32'
Created LLM model with id 'uKfkp5oB6Os6SYy-CQ18'
Created memory container with id 'uqfkp5oB6Os6SYy-CQ2t'
OpenSearch Agentic Memory Demo - Session: demo_session0
Type 'q' or 'quit' to end the conversation

👤 You: My name is Bob and I like swimming
🤖 Assistant: I'll store this information about you in memory so I can remember it for future conversations.
Tool #1: opensearch_memory
╭─────────────────────────────────────── Add long-term memory for session demo_sesison0, agent demo_agent ────────────────────────────────────────╮
│ User's name is Bob and he likes swimming                                                                                                │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────────────────────────────────────── Memory Stored ─────────────────────────────────────────────────────────────╮
│ ✅ Memory stored successfully:                                                                                                          │
│ 🔑 Memory ID: xKfkp5oB6Os6SYy-Nw2Q                                                                                                      │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Nice to meet you, Bob! I've stored that information so I'll remember that your name is Bob and that you enjoy swimming. Is there anything specific about swimming you'd like to talk about, or anything else you'd like me to remember about you?

👤 You: Do you know my name and hobby?
🤖 Assistant: Let me search my memory to recall what I know about you.
Tool #2: opensearch_memory
╭──────────────────────────────────────────────────────────── Search Results ─────────────────────────────────────────────────────────────╮
│                              Long term memories                                                                                         │
│ ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓                                                           │
│ ┃ Memory ID            ┃ Content                                            ┃                                                           │
│ ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩                                                           │
│ │ xqfkp5oB6Os6SYy-QA3H │ Bob likes swimming.                                │                                                           │
│ └──────────────────────┴────────────────────────────────────────────────────┘                                                           │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Yes, I do! Based on my memory, your name is **Bob** and your hobby is **swimming**. I remembered this from our earlier conversation where you introduced yourself to me.

👤 You: q
🤖 Assistant: Goodbye!


# New session
% python strands/strands_long_term.py
Find memory container with id 'uqfkp5oB6Os6SYy-CQ2t' by name 'demo_memory_container'
OpenSearch Agentic Memory Demo - Session: demo_session1
Type 'q' or 'quit' to end the conversation

👤 You: Now I don't like swimming any more, I like playing basketball now.
🤖 Assistant: I'll store this updated preference for you, noting that you now like playing basketball instead of swimming.
Tool #1: opensearch_memory
╭─────────────────────────────────────── Add long-term memory for session demo_session1, agent demo_agent ────────────────────────────────────────╮
│ User no longer likes swimming and now prefers playing basketball                                                                        │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────────────────────────────────────── Memory Stored ─────────────────────────────────────────────────────────────╮
│ ✅ Memory stored successfully:                                                                                                          │
│ 🔑 Memory ID: yKflp5oB6Os6SYy-Iw0L                                                                                                      │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Perfect! I've stored your updated preference. Your memory now shows that you no longer like swimming and have switched to enjoying basketball instead. This information is saved and I'll remember it for future conversations.

👤 You: Do you know my name and hobby?
🤖 Assistant: Let me search through your stored memories to find information about your name and hobby.
Tool #2: opensearch_memory
╭──────────────────────────────────────────────────────────── Search Results ─────────────────────────────────────────────────────────────╮
│                              Long term memories                                                                                         │
│ ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓                                                           │
│ ┃ Memory ID            ┃ Content                                            ┃                                                           │
│ ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩                                                           │
│ │ yaflp5oB6Os6SYy-Mg0g │ Bob no longer likes swimming and now prefers       │                                                           │
│ │                      │ playing basketball.                                │                                                           │
│ └──────────────────────┴────────────────────────────────────────────────────┘                                                           │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Based on the search results, I can see that I have information about you! According to my memories:

**Your name:** Bob

**Your current hobby:** You now like playing basketball (you previously liked swimming but changed your preference)

So yes, I know that your name is Bob and your current hobby is playing basketball. I also remember that you used to like swimming but switched to preferring basketball instead.

👤 You: q
🤖 Assistant: Goodbye!
```

## LangGraph (Short-term memory)

1. Configure environment

```bash
# Copy LangGraph environment template
cp langgraph/.env.langgraph-short.example langgraph/.env.langgraph-short

# Edit .env.langgraph-short file with the required configurations
```

> See langgraph/.env.langgraph-short.example for all available configuration options

2. Run the script

```bash
python langgraph/langgraph_short_term.py
```

The script creates an interactive chat session where you can converse with the AI agent. The agent maintains conversation history using OpenSearch's checkpoint storage, allowing it to remember the conversation context within the same thread.

- Enter your messages at the "You:" prompt
- Type `q` or `quit` to exit

**Example conversation:**

```
% python langgraph/langgraph_short_term.py
1. Setting up OpenSearch checkpointer...
✅ Created memory container: -adAuJoB6Os6SYy-bhFa

2. Creating chatbot...
✅ Chatbot ready

Starting new thread: demo_20251124_154342
LangGraph Interactive Demo
Type 'q' or 'quit' to end the conversation

👤 You: My name is Sarah and I love to bake
⚠️  No checkpoint found for thread_id=demo_20251124_154342, checkpoint_ns=, checkpoint_id=None
🤖 Assistant: That's wonderful that you enjoy baking, Sarah! Baking can be such a fun and creative outlet. What are some of your favorite things to bake? Do you have any specialties or recipes you're particularly proud of? I'd love to hear more about your baking passion.

👤 You: I like to bake a pie
🤖 Assistant: Ooh, pie baking is such a delicious art! There are so many wonderful varieties of pies to make. Do you have a favorite type of pie that you enjoy baking the most? Fruit pies like apple or cherry are classics, but cream pies and nut pies can be amazing too. 

I'd also love to hear about your pie crust techniques - do you make your own crust from scratch or do you have a preferred store-bought crust? Getting that perfect flaky crust is one of the biggest challenges of pie baking in my experience.

Please share any tips or secrets you have for making awesome pies! I'm always looking to improve my own baking skills, especially when it comes to pies. Bakers who love what they do always have the best insights.

👤 You: q
🤖 Assistant: Goodbye!

======================================================================

📊 Session Summary:
Thread ID: demo_20251124_154342
Total messages: 4
Session ended: 2025-11-24 15:44:16.845425

# Run script again with the same memory container name
% python langgraph/langgraph_short_term.py
1. Setting up OpenSearch checkpointer...
✅ Find memory container with id '-adAuJoB6Os6SYy-bhFa' by name 'langgraph_short_term'

2. Creating chatbot...
✅ Chatbot ready

Found existing thread: demo_20251124_154342
Resume existing conversation? (y/n): y
Resuming thread: demo_20251124_154342
LangGraph Interactive Demo
Type 'q' or 'quit' to end the conversation

👤 You: Do you know my name and hobby?
🤖 Assistant: Yes, you mentioned earlier that your name is Sarah and that you love to bake, specifically that you like baking pies.

👤 You: q
🤖 Assistant: Goodbye!

======================================================================

📊 Session Summary:
Thread ID: demo_20251124_154342
Total messages: 6
Session ended: 2025-11-24 15:44:37.339253
```

## LangGraph (Long-term memory)

1. Configure environment

```bash
# Copy LangGraph environment template
cp langgraph/.env.langgraph-long.example langgraph/.env.langgraph-long

# Edit .env.langgraph-long file with the required configurations
```

> See langgraph/.env.langgraph-long.example for all available configuration options


2. Run the script

```bash
python langgraph/langgraph_long_term.py
```

The script creates an interactive chat session with long-term memory capabilities. It maintains conversation context within sessions using OpenSearch checkpoints while also storing and retrieving important information across different sessions using semantic search-based long-term memory.

- Enter your messages at the "You:" prompt
- Type `q` or `quit` to exit

**Example conversation:**

```
% python langgraph/langgraph_long_term.py
Created embedding model with id 'HqchuJoB6Os6SYy-cxHw'
Created LLM model with id 'IKchuJoB6Os6SYy-dBFz'
Created memory container with id 'IachuJoB6Os6SYy-dBGq'

1. Setting up OpenSearch checkpointer...
✅ Use memory container with ID: IachuJoB6Os6SYy-dBGq

2. Creating chatbot...
✅ Chatbot ready

Starting new thread: demo_20251124_150952
LangGraph Interactive Demo
Type 'q' or 'quit' to end the conversation

👤 You: My name is Sarah and I love to bake
⚠️  No checkpoint found for thread_id=demo_20251124_150952, checkpoint_ns=, checkpoint_id=None
╭───────────────────────── Add long-term memory for session langgraph_long_term_session, agent langgraph_agent ─────────────────────────╮
│ User Sarah loves to bake                                                                                                                │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────────────────────────────────────── Memory Stored ─────────────────────────────────────────────────────────────╮
│ ✅ Memory stored successfully:                                                                                                          │
│ 🔑 Memory ID: N6chuJoB6Os6SYy-oRFt                                                                                                      │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
🤖 Assistant: Nice to meet you Sarah! That's wonderful that you enjoy baking. I love the aroma of freshly baked goods. What are some of your favorite things to bake? I'd be happy to share recipes or baking tips if you're interested.

👤 You: I like to bake a pie
╭───────────────────────── Add long-term memory for session langgraph_long_term_session, agent langgraph_agent ─────────────────────────╮
│ Sarah likes to bake pies                                                                                                                │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────────────────────────────────────── Memory Stored ─────────────────────────────────────────────────────────────╮
│ ✅ Memory stored successfully:                                                                                                          │
│ 🔑 Memory ID: R6chuJoB6Os6SYy-0xEb                                                                                                      │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
🤖 Assistant: Pies are delicious! I love a good fruit pie, especially apple or cherry. Do you have a favorite type of pie you like to bake? Baking pies can be an art with getting that perfect flaky crust. I'd love to hear any tips or secrets you've learned for making great pies. Baking is such a rewarding hobby.

👤 You: q
🤖 Assistant: Goodbye!

======================================================================

📊 Session Summary:
Thread ID: demo_20251124_150952
Total messages: 8
Session ended: 2025-11-24 15:10:26.668944

# New session
% python langgraph/langgraph_long_term.py
Find memory container with id 'IachuJoB6Os6SYy-dBGq' by name 'langgraph_long_term'

1. Setting up OpenSearch checkpointer...
✅ Use memory container with ID: IachuJoB6Os6SYy-dBGq

2. Creating chatbot...
✅ Chatbot ready

Found existing thread: demo_20251124_150952
Resume existing conversation? (y/n): y
Resuming thread: demo_20251124_150952
LangGraph Interactive Demo
Type 'q' or 'quit' to end the conversation

👤 You: Do you know my name and hobby?
╭──────────────────────────────────────────────────────────── Search Results ─────────────────────────────────────────────────────────────╮
│                              Long term memories                                                                                         │
│ ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓                                                           │
│ ┃ Memory ID            ┃ Content                                            ┃                                                           │
│ ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩                                                           │
│ │ P6chuJoB6Os6SYy-sxEW │ Sarah loves to bake, especially pies.              │                                                           │
│ └──────────────────────┴────────────────────────────────────────────────────┘                                                           │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
🤖 Assistant: Yes, from the memories I have stored, I know your name is Sarah and that you enjoy baking, especially pies. Baking pies is a wonderful hobby - there's something so satisfying about making a delicious pie from scratch with a flaky crust. Do you have a favorite type of pie you like to bake or a special family recipe? I'd love to hear more about your pie baking adventures!

👤 You: q
🤖 Assistant: Goodbye!

======================================================================

📊 Session Summary:
Thread ID: demo_20251124_150952
Total messages: 12
Session ended: 2025-11-24 15:12:09.996516
```