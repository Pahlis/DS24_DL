import os
from google import genai
import numpy as np
import json


class TextProcessor:
    """
    Ansvarar för all textbearbetning och skapande av embeddings (vektorrepresentationer av text).
    Denna klass tar hand om att:
    - Konvertera text till numeriska vektorer via Google's API
    - Dela upp långa texter i hanterbara delar (chunks)
    - Förbereda dokument för senare sökning och jämförelse
    
    Används som grund för RAG-systemet.  
    """
    
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
    
    def get_embedding(self, text):
        """Hämtar embedding för en text"""
        if not text or not text.strip():
            return None  # För att undvika API-anrop för tom text

        try:
            response = self.client.models.embed_content(
                model="text-embedding-004",
                contents=text
            )
            if hasattr(response, "embeddings"):
                return response.embeddings[0].values if response.embeddings else None
            elif hasattr(response, "embedding"):
                return response.embedding.values
            else:
                return None
        except Exception as e:
            raise Exception(f"Fel vid embedding: {e}")
    
    def chunk_text(self, text, chunk_size=500):
        """Delar upp text i chunks
        Strategin är att:
        1. Dela först på stycken (dubbla radbrytningar)
        2. Om stycke är för långt, dela på meningar
        3. Håll meningar tillsammans så länge som möjligt
        """
        paragraphs = text.split('\n\n')
        chunks = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if len(para) <= chunk_size:
                chunks.append(para)
            else:
                # Dela upp stycket i meningar
                sentences = para.split('. ')
                current_chunk = ""
                
                for sentence in sentences:
                    #Kontrollera att meningen slutar med punkt, utropstecken eller frågetecken
                    if sentence[-1] not in ".!?": 
                        sentence += "."

                    # Kontrollera om meningen är för lång annars starta på nästa chunk
                    if len(current_chunk + sentence) <= chunk_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
                
                #Lägg till sista chunk om den inte är tom
                if current_chunk:
                    chunks.append(current_chunk.strip())
        
        # Filtrera bort tomma chunks
        return [chunk for chunk in chunks if chunk.strip()]
    
    def prepare_document_embeddings(self, text_content, progress_callback=None):
        """
        Förbereder embeddings för en hel text genom att:
        1. Dela upp texten i chunks
        2. Skapa embeddings för varje chunk
        3. Returnera både embeddings och chunks för senare användning
        """
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
            else:
                print(f"Embedding saknas för chunk {i}")
                        
        return np.array(embeddings), valid_chunks


class FeedbackGenerator:
    """
    Klass för att generera feedback baserat på elevens återberättelse
    och originaltexten med hjälp av RAG (Retrieval-Augmented Generation).:
    Använder cosine similarity för att hitta de mest relevanta delarna
    av originaltexten för jämförelse med elevens återberättelse.
    """
    
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
    
    def cosine_similarity(self, vec1, vec2):
        """Beräknar cosine similarity mellan två vektorer"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def retrieve_relevant_chunks(self, query_text, document_embeddings_np, document_chunks, text_processor, top_k=5):
        """Hämtar de mest relevanta delarna av originaltexten baserat på elevens återberättelse"""
        query_embedding = text_processor.get_embedding(query_text)
        if not query_embedding:
            return []
        
        #Beräkna cosine similarity mellan query och varje dokumentchunk
        similarities = []
        for i, doc_emb in enumerate(document_embeddings_np):
            sim = self.cosine_similarity(query_embedding, doc_emb)
            similarities.append((sim, document_chunks[i], i))
        
        #Sortera efter similarity score, relevanta chunks först
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
        """Skapar prompt för RAG-systemet"""
        original_context_str = "\n".join(relevant_original_chunks)
        
        # Skapar konversationshistorik
        history_str = ""
        if conversation_history and len(conversation_history) > 0:
            recent_history = conversation_history[-4:]
            history_str = "\n--- Tidigare Konversation ---"
            for msg in recent_history:
                history_str += f"\n{msg['role'].capitalize()}: {msg['content'][:200]}"
            history_str += "\n---------------------------"
        
        prompt = f"""

    Du är en hjälpsam och positiv lärare som ger feedback på en elevs återberättelse av en text. 
    Eleven är 10 år och håller på att utveckla sin förmåga att återberätta.

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

    Din feedback ska börja med ett vänligt hej, men **utan att använda elevens namn**. Till exempel: "Hej!" eller "Hej där!"
    Jämför elevens text med originaltexten. 
    Fokusera på det viktigaste som eleven kan förbättra till nästa försök eller ge tydligt beröm om återberättelsen stämmer i stort. 
    Fokusera inte på för mycket på detaljer, utan ge en övergripande bedömning.
    
    Om återberättelsen innehåller stora delar originaltexten och bedöms vara förståelig återberättelse för en elev i årskurs 4-5: ge då ENDAST beröm och uppmuntran. 
    Om återberättelsen är välskriven, tydlig och innehåller de flesta viktiga detaljer, avsluta med "Fantastiskt jobb! Din återberättelse är verkligen klockren och du har fått med alla viktiga detaljer. Bra jobbat! [UPPGIFT_KLAR]". Se till att alltid inkludera minst en uppmuntrande mening innan [UPPGIFT_KLAR].
    Om återberättelsen är bra men saknar några detaljer eller har mindre språkliga fel, ge konstruktiv feedback och exempel på hur eleven kan förbättra sin text. Men skriv att eleven inte måste förbättra mer om den inte vill.
    Om återberättelsen är otydlig, saknar viktiga detaljer eller innehåller språkliga fel, ge konstruktiv feedback och exempel på hur eleven kan förbättra sin text.
    Fokusera på att ge tydliga, konkreta tips som eleven kan använda för att förbättra sin återberättelse nästa gång.
    Använd ett positivt och uppmuntrande språk, som om du är en engagerad lärare som vill hjälpa eleven att växa.

    Om du inte kan ge feedback, t.ex. om texten är för kort eller otydlig, svara med "Jag behöver mer information för att kunna ge feedback.".
    Din feedback:
    """

        return prompt
    
    def get_feedback(self, user_recollection, document_embeddings_np, document_chunks, text_processor, conversation_history):
        """Hämta feedback från Gemini, huvudmetoden för feedbackprocessen"""
        if document_embeddings_np is None or len(document_embeddings_np) == 0:
            return "Vänligen välj en text att återberätta först."
        
        #Hämta relevanta chunks från originaltexten
        relevant_chunks = self.retrieve_relevant_chunks(
            user_recollection, document_embeddings_np, document_chunks, text_processor, top_k=10
        )
        if not relevant_chunks:
            return "Kunde inte hitta relevant information från texten."
        
        #Skapa RAG-prompten och få feedback
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
        """Laddar texter från JSON-fil"""
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    
    @staticmethod
    def create_custom_text_info(custom_text):
        """Skapar textinfo för egen text"""
        return {
            'title': 'Din egen text',
            'subject': 'Egen',
            'level': 'Anpassad',
            'content': custom_text.strip()
        }
    
    @staticmethod
    def generate_text_key(text_info, index=None):
        """Genererar unik nyckel för text"""
        if text_info['subject'] == 'Egen':
            return f"custom_{hash(text_info['content'])}"
        else:
            return f"predefined_{index}"

class ChatbotEvaluator:
    """ 
    Ansvarar för att utvärdera chatbotens svar på elevens återberättelse.
    Denna klass skickar chatbotens svar till LLM för att få en score samt en motivering.
    Använder Google Gemini API för att generera en bedömning baserat på tydlighet, relevans och pedagogiskt värde.    
    """
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
        
        # Letar efter totalbetyg
        total_match = re.search(r"TOTALBETYG:\s*(\d+)", evaluation_text)
        if total_match:
            return int(total_match.group(1))
        
        # Fallback: letar efter första nummer/10
        score_match = re.search(r"(\d+)/10", evaluation_text)
        if score_match:
            return int(score_match.group(1))
        
        return None

    def get_evaluation_summary(self, user_recollection, chatbot_response, original_text=None):
        """För både fullständig utvärdering och extraherad score"""
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
        """Ställer in API-nyckel från secrets"""
        try:
            return secrets["GEMINI_API_KEY"]
        except KeyError:
            raise Exception("GEMINI_API_KEY kunde inte hittas i secrets")
    
    def process_text_selection(self, text_info, progress_callback=None):
        """Bearbetar vald text och skapa embeddings"""
        return self.text_processor.prepare_document_embeddings(
            text_info['content'], 
            progress_callback
        )
    
    def generate_feedback(self, user_input, embeddings, chunks, conversation_history):
        """Genererar feedback för användarens input"""
        return self.feedback_generator.get_feedback(
            user_input, 
            embeddings, 
            chunks, 
            self.text_processor,
            conversation_history
        )
    
    def load_available_texts(self):
        """Laddar tillgängliga texter"""
        return self.text_manager.load_texts()
    
    def create_custom_text(self, text_content):
        """Skapar custom text info"""
        return self.text_manager.create_custom_text_info(text_content)
    
    def generate_text_key(self, text_info, index=None):
        """Genererar text key"""
        return self.text_manager.generate_text_key(text_info, index)
    
