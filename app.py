import os
import io
import asyncio
import aiohttp
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import pytesseract
from PIL import Image
from PyPDF2 import PdfReader
import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import time
import json

load_dotenv()

@dataclass
class ProcessedDocument:
    """Structure for processed document information"""
    filename: str
    content: str
    chunks: List[str]
    total_chars: int
    doc_type: str
    summary: str = ""

@dataclass
class AgentResponse:
    """Structure for storing agent responses"""
    agent_name: str
    content: str
    confidence: float
    metadata: Dict = None
    processing_time: float = 0.0

class DeploymentDocumentProcessor(DocumentProcessor):
    """Enhanced document processor with deployment-specific FAISS handling"""
    def __init__(self, index_path: str = "deployment/faiss_index"):
        super().__init__()
        self.index_path = index_path
        self.vector_store = None
        self.index_metadata = {}
        self._ensure_deployment_directory()

    def _ensure_deployment_directory(self):
        """Ensure the deployment directory exists"""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

    def _save_index_metadata(self):
        """Save metadata about the index"""
        metadata_path = f"{self.index_path}_metadata.pkl"
        metadata = {
            'last_updated': datetime.now().isoformat(),
            'total_vectors': self.vector_store.index.ntotal if self.vector_store else 0,
            'dimension': self.vector_store.index.d if self.vector_store else 0
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

    def _load_index_metadata(self) -> Dict:
        """Load index metadata"""
        metadata_path = f"{self.index_path}_metadata.pkl"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                return pickle.load(f)
        return {}

    async def update_vector_store(self, documents: List[ProcessedDocument], progress_callback) -> bool:
        try:
            all_chunks = []
            metadata_list = []
            
            for idx, doc in enumerate(documents):
                progress_callback(0.2 + (0.6 * (idx / len(documents))), 
                                f"Indexing {doc.filename}")
                
                for chunk_idx, chunk in enumerate(doc.chunks):
                    all_chunks.append(chunk)
                    metadata_list.append({
                        "source": doc.filename,
                        "chunk_index": chunk_idx,
                        "doc_type": doc.doc_type,
                        "timestamp": datetime.now().isoformat()
                    })

            if all_chunks:
                progress_callback(0.8, "Creating deployment index")
                
                # Get embeddings for all chunks
                embeddings = await self._get_embeddings_batch(all_chunks)
                
                # Initialize FAISS index
                dimension = len(embeddings[0])
                index = faiss.IndexFlatL2(dimension)
                
                # Add vectors to the index
                index.add(np.array(embeddings))
                
                # Save the index
                faiss.write_index(index, f"{self.index_path}.faiss")
                
                # Save metadata separately
                with open(f"{self.index_path}_meta.pkl", 'wb') as f:
                    pickle.dump({
                        'metadata': metadata_list,
                        'chunks': all_chunks
                    }, f)
                
                # Update the vector store
                self.vector_store = self._load_vector_store()
                
                progress_callback(1.0, "Deployment index ready")
                return True
                
        except Exception as e:
            st.error(f"Deployment index update error: {str(e)}")
            return False

    async def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts"""
        try:
            embeddings = []
            batch_size = 10  # Adjust based on your needs
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = await asyncio.gather(*[
                    self.embeddings.aembed_query(text) for text in batch
                ])
                embeddings.extend(batch_embeddings)
            
            return embeddings
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")

    def _load_vector_store(self) -> Optional[FAISS]:
        """Load the vector store from saved index"""
        try:
            if os.path.exists(f"{self.index_path}.faiss"):
                # Load the FAISS index
                index = faiss.read_index(f"{self.index_path}.faiss")
                
                # Load metadata and chunks
                with open(f"{self.index_path}_meta.pkl", 'rb') as f:
                    data = pickle.load(f)
                
                # Create new FAISS vector store
                vector_store = FAISS(
                    self.embeddings.embed_query,
                    index,
                    data['chunks'],
                    data['metadata']
                )
                
                return vector_store
            return None
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            return None

    async def similarity_search(self, query: str, k: int = 3) -> List[Dict]:
        """Perform similarity search with error handling"""
        try:
            if not self.vector_store:
                self.vector_store = self._load_vector_store()
                if not self.vector_store:
                    raise Exception("No vector store available")
            
            # Get query embedding
            query_embedding = await self.embeddings.aembed_query(query)
            
            # Perform search
            D, I = self.vector_store.index.search(
                np.array([query_embedding]), k
            )
            
            results = []
            for i, idx in enumerate(I[0]):
                if idx < len(self.vector_store.docstore._dict):
                    doc = self.vector_store.docstore._dict[idx]
                    results.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'score': float(D[0][i])
                    })
            
            return results
            
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return []

class AgentStatus:
    """Enhanced agent status management with sidebar display"""
    def __init__(self):
        self.sidebar_placeholder = None
        self.agents = {
            'document_processor': {'status': 'idle', 'progress': 0, 'message': ''},
            'threat_hunter': {'status': 'idle', 'progress': 0, 'message': ''},
            'vulnerability_analyst': {'status': 'idle', 'progress': 0, 'message': ''},
            'incident_responder': {'status': 'idle', 'progress': 0, 'message': ''},
            'compliance_advisor': {'status': 'idle', 'progress': 0, 'message': ''}
        }
        
    def initialize_sidebar_placeholder(self):
        """Initialize the sidebar placeholder"""
        with st.sidebar:
            self.sidebar_placeholder = st.empty()
    
    def update_status(self, agent_name: str, status: str, progress: float, message: str = ""):
        """Update agent status and refresh sidebar display"""
        self.agents[agent_name] = {
            'status': status,
            'progress': progress,
            'message': message
        }
        self._render_status()

    def _render_status(self):
        """Render status in sidebar"""
        if self.sidebar_placeholder is None:
            self.initialize_sidebar_placeholder()
            
        with self.sidebar_placeholder.container():
            for agent_name, status in self.agents.items():
                self._render_agent_card(agent_name, status)

    def _render_agent_card(self, agent_name: str, status: dict):
        """Render individual agent status card in sidebar"""
        colors = {
            'idle': '#6c757d',
            'working': '#007bff',
            'completed': '#28a745',
            'error': '#dc3545'
        }
        color = colors.get(status['status'], colors['idle'])
        
        st.markdown(f"""
            <div style="
                background-color: #1E1E1E;
                padding: 0.8rem;
                border-radius: 0.5rem;
                margin-bottom: 0.8rem;
                border: 1px solid {color};
            ">
                <div style="color: {color}; font-weight: bold;">
                    {agent_name.replace('_', ' ').title()}
                </div>
                <div style="
                    color: #CCCCCC;
                    font-size: 0.8rem;
                    margin: 0.3rem 0;
                ">
                    {status['message'] or status['status'].title()}
                </div>
                <div style="
                    height: 4px;
                    background-color: rgba(255,255,255,0.1);
                    border-radius: 2px;
                    margin-top: 0.5rem;
                ">
                    <div style="
                        width: {status['progress'] * 100}%;
                        height: 100%;
                        background-color: {color};
                        border-radius: 2px;
                        transition: width 0.3s ease;
                    "></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

class CyberSecurityAgent:
    """Cybersecurity expert system with specialized agents"""
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.3,
            model_name="llama-3.1-70b-versatile",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        self.chat_history = []
        self.doc_processor = DocumentProcessor()
        self._initialize_prompts()
        self.agents = self._initialize_agents()

    def _initialize_prompts(self):
        """Initialize specialized cybersecurity agent prompts"""
        self.prompts = {
            'threat_hunter': """You are an advanced Threat Hunter AI.
Context: {context}
Query: {query}
Chat History: {chat_history}

Analyze for:
1. Potential threats and IOCs
2. Attack patterns
3. Recommended monitoring
Keep response focused and actionable.""",

            'vulnerability_analyst': """You are a Vulnerability Analysis AI.
Context: {context}
Query: {query}
Chat History: {chat_history}

Provide:
1. Vulnerability assessment
2. Risk rating
3. Mitigation steps
Be specific and practical.""",

            'incident_responder': """You are an Incident Response AI.
Context: {context}
Query: {query}
Chat History: {chat_history}

Focus on:
1. Immediate response actions
2. Containment strategies
3. Recovery steps
Prioritize critical responses.""",

            'compliance_advisor': """You are a Security Compliance AI.
Context: {context}
Query: {query}
Chat History: {chat_history}

Address:
1. Compliance requirements
2. Policy implications
3. Documentation needs
Ensure regulatory alignment."""
        }

    def _initialize_agents(self):
        return {
            name: ChatPromptTemplate.from_messages([
                ("system", prompt),
                ("human", "{input}")
            ]) | self.llm | StrOutputParser()
            for name, prompt in self.prompts.items()
        }

    def _format_chat_history(self) -> str:
        formatted = []
        for msg in self.chat_history[-5:]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            formatted.append(f"{role}: {msg.content}")
        return "\n".join(formatted)

    async def process_documents(self, files, status_callback) -> bool:
        try:
            processed_docs = []
            
            for idx, file in enumerate(files):
                doc = await self.doc_processor.process_file(
                    file,
                    lambda p, m: status_callback(
                        'document_processor',
                        'working',
                        (idx / len(files)) + (p / len(files)),
                        m
                    )
                )
                if doc:
                    processed_docs.append(doc)

            if processed_docs:
                success = await self.doc_processor.update_vector_store(
                    processed_docs,
                    lambda p, m: status_callback(
                        'document_processor',
                        'working',
                        0.8 + (p * 0.2),
                        m
                    )
                )
                
                if success:
                    status_callback(
                        'document_processor',
                        'completed',
                        1.0,
                        "Security analysis complete"
                    )
                    return True

            status_callback(
                'document_processor',
                'error',
                0,
                "Document processing failed"
            )
            return False
            
        except Exception as e:
            status_callback(
                'document_processor',
                'error',
                0,
                str(e)
            )
            return False

    async def get_relevant_context(self, query: str) -> str:
        try:
            results = await self.doc_processor.similarity_search(query, k=3)
            if results:
                return "\n\n".join(result['content'] for result in results)
        except Exception as e:
            st.error(f"Error retrieving context: {str(e)}")
        return ""

    async def process_query(
        self,
        query: str,
        status_callback
    ) -> Dict[str, AgentResponse]:
        responses = {}
        context = await self.get_relevant_context(query)
        chat_history = self._format_chat_history()
        
        try:
            agent_tasks = []
            for agent_name in ['threat_hunter', 'vulnerability_analyst', 
                             'incident_responder', 'compliance_advisor']:
                status_callback(agent_name, 'working', 0.2, f"Analyzing security context")
                agent_tasks.append(self._get_agent_response(
                    agent_name, query, context, chat_history
                ))

            agent_responses = await asyncio.gather(*agent_tasks)
            
            for agent_name, response in zip(
                ['threat_hunter', 'vulnerability_analyst', 
                 'incident_responder', 'compliance_advisor'],
                agent_responses
            ):
                responses[agent_name] = response
                status_callback(
                    agent_name,
                    'completed',
                    1.0,
                    f"{agent_name.replace('_', ' ').title()} analysis complete"
                )

            final_response = await self._synthesize_security_responses(
                query, context, chat_history, responses
            )
            responses['final_analysis'] = final_response

            self.chat_history.extend([
                HumanMessage(content=query),
                AIMessage(content=final_response.content)
            ])

            return responses

        except Exception as e:
            for agent in self.agents.keys():
                status_callback(agent, 'error', 0, str(e))
            raise Exception(f"Security analysis error: {str(e)}")

    async def _get_agent_response(
        self,
        agent_name: str,
        query: str,
        context: str,
        chat_history: str
    ) -> AgentResponse:
        """Get response from specific security agent with metadata"""
        start_time = time.time()
        
        try:
            response = await self.agents[agent_name].ainvoke({
                "input": query,
                "context": context,
                "query": query,
                "chat_history": chat_history
            })
            
            processing_time = time.time() - start_time
            
            metadata = {
                "processing_time": processing_time,
                "context_length": len(context),
                "query_length": len(query),
                "security_confidence": self._calculate_security_confidence(response)
            }
            
            return AgentResponse(
                agent_name=agent_name,
                content=response,
                confidence=self._calculate_security_confidence(response),
                metadata=metadata,
                processing_time=processing_time
            )
            
        except Exception as e:
            raise Exception(f"Security agent {agent_name} error: {str(e)}")

    def _calculate_security_confidence(self, response: str) -> float:
        """Calculate confidence score based on security-specific indicators"""
        confidence = 0.7  # Base confidence
        
        # Increase confidence based on security-specific indicators
        security_indicators = [
            "CVE-", "CVSS", "vulnerability", "threat", "attack",
            "exploit", "patch", "mitigation", "compliance"
        ]
        
        for indicator in security_indicators:
            if indicator.lower() in response.lower():
                confidence += 0.03
                
        return min(0.95, confidence)  # Cap at 0.95

    async def _synthesize_security_responses(
        self,
        query: str,
        context: str,
        chat_history: str,
        responses: Dict[str, AgentResponse]
    ) -> AgentResponse:
        """Synthesize final security analysis from all agent responses"""
        try:

            greetings = ['hi', 'hello', 'hey', 'hii', 'greetings']
            if query.lower().strip() in greetings:
                return AgentResponse(
                    agent_name="final_analysis",
                    content="Hello! How can I assist you with your security needs today?",
                    confidence=1.0,
                    metadata={"greeting": True},
                    processing_time=0.1
                )

            formatted_responses = "\n\n".join([
                f"{name.upper()}:\n{response.content}"
                for name, response in responses.items()
                if name != 'final_analysis'
            ])

            start_time = time.time()
            
            synthesis_template = """
            Analyze and synthesize the following security assessments:
            {formatted_responses}
            
            Provide a consolidated security analysis that includes:
            1. Critical findings and threats
            2. Recommended actions
            3. Compliance implications
            
            Keep the response clear and actionable.
            """

            
            synthesis_prompt = ChatPromptTemplate.from_messages([
                ("system", synthesis_template)
            ])
            
            synthesis_chain = synthesis_prompt | self.llm | StrOutputParser()
            
            synthesis_response = await synthesis_chain.ainvoke({
                "formatted_responses": formatted_responses
            })
            
            processing_time = time.time() - start_time
            
            # Calculate overall confidence based on individual agent confidences
            overall_confidence = sum(
                response.confidence for response in responses.values()
            ) / len(responses)
            
            metadata = {
                "processing_time": processing_time,
                "source_responses": len(responses),
                "context_used": bool(context),
                "overall_confidence": overall_confidence
            }
            
            return AgentResponse(
                agent_name="final_analysis",
                content=synthesis_response,
                confidence=overall_confidence,
                metadata=metadata,
                processing_time=processing_time
            )

        except Exception as e:
            raise Exception(f"Security synthesis error: {str(e)}")

def setup_streamlit_ui():
    st.set_page_config(
        page_title="Cybersecurity Expert System",
        page_icon="🛡️",
        layout="wide"
    )
    
    st.markdown("""
        <style>
        /* Add these CSS rules */
        [data-testid="stSidebar"] {
            background-color: #0a192f;
        }
        
        /* Style for sidebar slider */
        [data-testid="stSidebarNav"] {
            background-color: #0a192f;
        }

        /* Customize file uploader */
        .uploadedFile {
            background-color: rgba(100, 255, 218, 0.05);
            border-left: 3px solid #64ffda;
            padding: 0.5rem;
            margin: 0.5rem 0;
        }

        /* Existing styles stay the same */
        .stApp {
            background-color: #0a192f;
            color: #8892b0;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            border: 1px solid #64ffda;
            background-color: rgba(100, 255, 218, 0.1);
            font-family: 'Courier New', monospace;
        }
        .agent-card {
            padding: 1rem;
            margin-bottom: 1rem;
            border: 1px solid #64ffda;
            border-radius: 0.5rem;
            background-color: rgba(100, 255, 218, 0.05);
        }
        .metadata-section {
            font-size: 0.8rem;
            color: #64ffda;
            margin-top: 0.5rem;
        }
        .security-status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin: 0.5rem 0;
        }
        .security-icon {
            font-size: 1.5rem;
            color: #64ffda;
        }
        
        /* Style scrollbars */
        ::-webkit-scrollbar {
            width: 10px;
            background: #0a192f;
        }
        ::-webkit-scrollbar-track {
            background: #0a192f;
        }
        ::-webkit-scrollbar-thumb {
            background: #64ffda;
            border-radius: 5px;
        }
        
        /* Style file uploader button */
        .stButton > button {
            background-color: rgba(100, 255, 218, 0.1);
            color: #64ffda;
            border: 1px solid #64ffda;
        }
        .stButton > button:hover {
            background-color: rgba(100, 255, 218, 0.2);
            color: #64ffda;
            border: 1px solid #64ffda;
        }
        </style>
        <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    """, unsafe_allow_html=True)

def main():
    """Main application with cybersecurity focus"""
    setup_streamlit_ui()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = CyberSecurityAgent()
    if "agent_status" not in st.session_state:
        st.session_state.agent_status = AgentStatus()
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
    
    st.markdown("""
        <div style='text-align: center; color: #64ffda;'>
            <h1>🛡️ Cybersecurity Expert System</h1>
        </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("""
            <div style='color: #64ffda;'>
                <h3>📋 Security Document Analysis</h3>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload security documents (PDF/Images)",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key="document_uploader"
        )
        
        if uploaded_files:
            st.markdown("""
                <div style='color: #64ffda;'>
                    <h4>📎 Selected Documents</h4>
                </div>
            """, unsafe_allow_html=True)
            
            for file in uploaded_files:
                st.markdown(f"""
                    <div class="uploadedFile">
                        <div style="color: #64ffda;">📄 {file.name}</div>
                        <div style="color: #8892b0; font-size: 0.8rem;">
                            Type: {file.type}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            if st.button("🔄 Analyze Security Documents", key="process_docs"):
                with st.spinner("Analyzing security documents..."):
                    async def process_docs():
                        await st.session_state.agent.process_documents(
                            uploaded_files,
                            st.session_state.agent_status.update_status
                        )
                        st.session_state.documents_processed = True
                    
                    asyncio.run(process_docs())
        
        st.markdown("""
            <div style='color: #64ffda;'>
                <h3>🤖 Security Agents Status</h3>
            </div>
        """, unsafe_allow_html=True)
        st.session_state.agent_status.initialize_sidebar_placeholder()
    
    st.markdown("### 💬 Security Analysis Interface")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], dict):
                st.markdown(f"""
                    <div class="chat-message {message['role']}">
                        {message['content']['final_analysis'].content}
                    </div>
                """, unsafe_allow_html=True)
                
                with st.expander("🔍 Detailed Security Analysis", expanded=False):
                    for agent_name, response in message['content'].items():
                        if agent_name != 'final_analysis':
                            st.markdown(f"""
                                <div class="agent-card">
                                    <div class="security-status">
                                        <span class="material-icons security-icon">
                                            {agent_name.split('_')[0]}
                                        </span>
                                        <strong>{agent_name.replace('_', ' ').title()}</strong>
                                    </div>
                                    <div style="margin: 0.5rem 0;">
                                        {response.content}
                                    </div>
                                    <div class="metadata-section">
                                        Confidence: {response.confidence:.2%}
                                        <br>
                                        Processing time: {response.processing_time:.2f}s
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-message {message['role']}">
                        {message['content']}
                    </div>
                """, unsafe_allow_html=True)
    
    if prompt := st.chat_input("Enter your security query..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            try:
                async def process_query():
                    return await st.session_state.agent.process_query(
                        prompt,
                        st.session_state.agent_status.update_status
                    )
                
                responses = asyncio.run(process_query())
                
                if responses:
                    response_placeholder.markdown(f"""
                        <div class="chat-message assistant">
                            {responses['final_analysis'].content}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": responses
                    })
                
            except Exception as e:
                response_placeholder.error(f"Security analysis error: {str(e)}")

if __name__ == "__main__":
    main()
