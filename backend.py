import os
from google import genai
import numpy as np
import json


class TextProcessor:
    """Hanterar textbearbetning och embeddings"""
    
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
    
    def get_embedding(self, text):
        """Hämta embedding för en text"""
        try:
            response = self.client.models.embed_content(
                model="text-embedding-004",
                contents=text
            )
            if hasattr(response, 'embeddings'):
                return response.embeddings[0].values if response.embeddings else None
            elif hasattr(response, 'embedding'):
                return response.embedding.values
            else:
                return None
        except Exception as e:
            raise Exception(f"Fel vid embedding: {e}")
    
    def chunk_text(self, text, chunk_size=500):
        """Dela upp text i chunks"""
        paragraphs = text.split('\n\n')
        chunks = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if len(para) <= chunk_size:
                chunks.append(para)
            else:
                sentences = para.split('. ')
                current_chunk = ""
                
                for sentence in sentences:
                    if not sentence.endswith('.') and sentence != sentences[-1]:
                        sentence += '.'
                        
                    if len(current_chunk + sentence) <= chunk_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def prepare_document_embeddings(self, text_content, progress_callback=None):
        """Förbered embeddings för en text"""
        chunks = self.chunk_text(text_content)
        embeddings = []
        valid_chunks = []
        
        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(i + 1, len(chunks))
            
            embedding = self.get_embedding(chunk)
            if embedding:
                embeddings.append(embedding)
                valid_chunks.append(chunk)
        
        return np.array(embeddings), valid_chunks


class FeedbackGenerator:
    """Hanterar feedback-generering med RAG"""
    
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
    
    def cosine_similarity(self, vec1, vec2):
        """Beräkna cosine similarity mellan två vektorer"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def retrieve_relevant_chunks(self, query_text, document_embeddings_np, document_chunks, text_processor, top_k=5):
        """Hämta relevanta chunks baserat på query"""
        query_embedding = text_processor.get_embedding(query_text)
        if not query_embedding:
            return []
        
        similarities = []
        for i, doc_emb in enumerate(document_embeddings_np):
            sim = self.cosine_similarity(query_embedding, doc_emb)
            similarities.append((sim, document_chunks[i], i))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Ta de mest relevanta chunks och lägg till lite extra kontext
        selected_chunks = []
        selected_indices = set()
        
        for sim, chunk, idx in similarities[:top_k]:
            if sim > 0.3:  # Endast chunks med rimlig relevans
                selected_chunks.append(chunk)
                selected_indices.add(idx)
        
        # Om vi har för få chunks, lägg till närliggande chunks för kontext
        if len(selected_chunks) < 3:
            for sim, chunk, idx in similarities:
                if len(selected_chunks) >= 5:
                    break
                if idx not in selected_indices:
                    selected_chunks.append(chunk)
                    selected_indices.add(idx)
        
        return selected_chunks
    
    def create_rag_prompt(self, user_recollection, relevant_original_chunks, conversation_history):
        """Skapa prompt för RAG-systemet"""
        original_context_str = "\n".join(relevant_original_chunks)
        
        # Skapa konversationshistorik
        history_str = ""
        if conversation_history and len(conversation_history) > 0:
            recent_history = conversation_history[-4:]
            history_str = "\n--- Tidigare Konversation ---"
            for msg in recent_history:
                history_str += f"\n{msg['role'].capitalize()}: {msg['content'][:300]}"
            history_str += "\n---------------------------"
        
        prompt = f"""

    Du är en hjälpsam och positiv lärare som ger feedback på en elevs återberättelse av en text. 
    Eleven går i årskurs 4-5 och håller på att utveckla sin förmåga att återberätta.

    **Dina feedbackkriterier (baserade på läroplanen):**
    -   **Innehåll:** Hur väl eleven har fått med huvudbudskapet och viktiga detaljer från originaltexten.
    -   **Språkliga strukturer:** Meningsbyggnad (varierad användning av huvudsatser och bisatser), korrekt användning av skiljetecken.
    -   **Textbindning:** Användning av sambandsord för att skapa sammanhang och flöde mellan meningar och stycken.
    -   **Språkliga normer:** Stavning, korrekt böjning av ord och användning av ordklasser.

    Här är tidigare dialog med eleven:
    {history_str}

    Här är originaltexten:
    {original_context_str}

    Här är elevens senaste återberättelse:
    {user_recollection}

    Jämför elevens text med originaltexten och ge uppmuntrande och handlingsorienterad feedback baserat på ovanstående kriterier. Max 10 meningar.
    Fokusera på det viktigaste som eleven kan förbättra till nästa försök, eller ge tydligt beröm om återberättelsen är komplett. Fokusera inte på för mycket på detaljer, utan ge en övergripande bedömning.
    Om återberättelsen är mycket kort (mindre än 50% av originaltexten) eller innehåller allvarliga brister, ge konstruktiv kritik och tips på hur eleven kan förbättra sin text.

    Om återberättelsen innehåller 70% av originaltexten och är välskriven (du bedömer detta utifrån originaltexten och elevens nivå), ge då endast beröm och uppmuntran, t.ex. "Fantastiskt jobb! Din återberättelse är verkligen klockren och du har fått med alla viktiga detaljer. Bra jobbat! Nu är du klar med den här uppgiften. [UPPGIFT_KLAR]". Se till att alltid inkludera minst en uppmuntrande mening innan [UPPGIFT_KLAR].
    Om återberättelsen behöver förbättras, ge konstruktiv feedback och tips på hur eleven kan förbättra sin text och ge gärna exempel, fokusera på senaste återberättelsen men uppmärksamma gärna viktiga ändringar i innehåll OCH EN språklig aspekt (t.ex. meningsbyggnad, skiljetecken, stavning, textbindning) glöm inte förklara vad aspekten innebär.
    Formulera gärna feedbacken som frågor om texten, exempelvis "Vad tror du att vikingarna handlade med?"
    Om du inte kan ge feedback, t.ex. om texten är för kort eller otydlig, svara med "Jag behöver mer information för att kunna ge feedback.".
    Din feedback:
    """

        return prompt
    
    def get_feedback(self, user_recollection, document_embeddings_np, document_chunks, text_processor, conversation_history):
        """Hämta feedback från Gemini"""
        if document_embeddings_np is None or len(document_embeddings_np) == 0:
            return "Vänligen välj en text att återberätta först."
        
        relevant_chunks = self.retrieve_relevant_chunks(
            user_recollection, document_embeddings_np, document_chunks, text_processor, top_k=10
        )
        if not relevant_chunks:
            return "Kunde inte hitta relevant information från texten."
        
        rag_prompt = self.create_rag_prompt(user_recollection, relevant_chunks, conversation_history)
        
        try:
            response = self.client.models.generate_content(
                model="gemini-1.5-flash",
                contents=[rag_prompt]
            )
            return response.text
        except Exception as e:
            return f"Tyvärr, något gick fel när jag skulle generera feedback: {e}"


class TextManager:
    """Hanterar textladdning och hantering"""
    
    @staticmethod
    def load_texts(filename="texts.json"):
        """Ladda texter från JSON-fil"""
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    @staticmethod
    def create_custom_text_info(custom_text):
        """Skapa textinfo för egen text"""
        return {
            'title': 'Din egen text',
            'subject': 'Egen',
            'level': 'Anpassad',
            'content': custom_text.strip()
        }
    
    @staticmethod
    def generate_text_key(text_info, index=None):
        """Generera unik nyckel för text"""
        if text_info['subject'] == 'Egen':
            return f"custom_{hash(text_info['content'])}"
        else:
            return f"predefined_{index}"

class ChatbotEvaluator:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)

    def evaluate_response(self, user_recollection, chatbot_response):
        """Skickar chatbotens svar till LLM för att få en score"""
        evaluation_prompt = f"""
        Du är en erfaren språklärare som utvärderar en AI-chattbots återkoppling till en elev. Bedöm chatbotens respons utifrån följande kriterier:

        1. Klarhet: Är svaret tydligt och begripligt?
        2. Relevans: Ger chatboten feedback som är relevant baserat på elevens återberättelse?
        3. Pedagogiskt värde: Är svaret engagerande och lärorikt för en elev i årskurs 4–5?

        Här är elevens återberättelse:
        {user_recollection}

        Här är chatbotens feedback:
        {chatbot_response}

        Ge en numerisk bedömning från **1 till 10**, samt en kort motivering.
        """
        try:
            response = self.client.models.generate_content(
                model="gemini-1.5-flash",
                contents=[evaluation_prompt]
            )
            return response.text
        except Exception as e:
            return f"Fel vid utvärdering: {e}"
        
    def parse_evaluation_score(self, evaluation_text):
        """Extrahera numerisk score från utvärderingstext"""
        import re
        
        # Leta efter totalbetyg
        total_match = re.search(r'TOTALBETYG:\s*(\d+)', evaluation_text)
        if total_match:
            return int(total_match.group(1))
        
        # Fallback: leta efter första nummer/10
        score_match = re.search(r'(\d+)/10', evaluation_text)
        if score_match:
            return int(score_match.group(1))
        
        return None

    def get_evaluation_summary(self, user_recollection, chatbot_response, original_text=None):
        """Få både fullständig utvärdering och extraherad score"""
        full_evaluation = self.evaluate_response(user_recollection, chatbot_response, original_text)
        score = self.parse_evaluation_score(full_evaluation)
        
        return {
            'full_evaluation': full_evaluation,
            'score': score,
            'timestamp': __import__('datetime').datetime.now().isoformat()
        }   
    
class AppBackend:
    """Huvudbackend-klass som koordinerar alla komponenter"""
    
    def __init__(self, api_key):
        self.text_processor = TextProcessor(api_key)
        self.feedback_generator = FeedbackGenerator(api_key)
        self.text_manager = TextManager()
        self.evaluator = ChatbotEvaluator(api_key) 
    
    def setup_api_key(self, secrets):
        """Ställ in API-nyckel från secrets"""
        try:
            return secrets["GEMINI_API_KEY"]
        except KeyError:
            raise Exception("GEMINI_API_KEY kunde inte hittas i secrets")
    
    def process_text_selection(self, text_info, progress_callback=None):
        """Bearbeta vald text och skapa embeddings"""
        return self.text_processor.prepare_document_embeddings(
            text_info['content'], 
            progress_callback
        )
    
    def generate_feedback(self, user_input, embeddings, chunks, conversation_history):
        """Generera feedback för användarens input"""
        return self.feedback_generator.get_feedback(
            user_input, 
            embeddings, 
            chunks, 
            self.text_processor,
            conversation_history
        )
    
    def load_available_texts(self):
        """Ladda tillgängliga texter"""
        return self.text_manager.load_texts()
    
    def create_custom_text(self, text_content):
        """Skapa custom text info"""
        return self.text_manager.create_custom_text_info(text_content)
    
    def generate_text_key(self, text_info, index=None):
        """Generera text key"""
        return self.text_manager.generate_text_key(text_info, index)



# Test för att se att importen fungerar
if __name__ == "__main__":
    print("Backend module loaded successfully!")
    print("Available classes:", [cls for cls in dir() if cls[0].isupper()])