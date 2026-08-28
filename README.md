# RAG : Retreival Augemented Generation
RAG is an AI framework that improves the accuracy and reliability of Large Language Models (LLMs) by fetching relevant facts from an external knowledge base before generating a response. 

### Core Components
- <b>Retrieval</b> : Seaching a database to find trusted context matching the user's query.
- <b>Augmentation</b> : Appending that retrieved context directly into the prompt given to the LLM.
- <b>Generation</b> : The LLM producing a response based strictly on the provided facts rather than relying solely on its training data.

### Main Benefits
- Reduces Hallucination
- Allows AI to access new information without expensive training
- Enables the system to cite specific documents or links for its answers