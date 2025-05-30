import streamlit as st
import sys
import os

# Lägg till current directory till Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from backend import AppBackend
except ImportError as e:
    st.error(f"Kunde inte importera backend: {e}")
    st.error("Se till att backend.py finns i samma mapp som app.py")
    st.stop()


# Konfiguration för sidan
st.set_page_config(
    page_title="Återberättelsecoachen",
    layout="wide"
)

# CSS för snyggare utseende
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.feedback-box {
    background-color: #f0f8ff;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initiera session state variabler"""
    if 'backend' not in st.session_state:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
            st.session_state.backend = AppBackend(api_key)
        except KeyError:
            st.error("Fel: GEMINI_API_KEY kunde inte hittas. Se till att den är satt i .streamlit/secrets.toml lokalt eller som secret i Streamlit Cloud.")
            st.stop()
        except Exception as e:
            st.error(f"Fel vid initialisering: {e}")
            st.stop()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'current_text_key' not in st.session_state:
        st.session_state.current_text_key = None
    
    if 'document_data' not in st.session_state:
        st.session_state.document_data = None
    
    if 'current_text_info' not in st.session_state:
        st.session_state.current_text_info = None
    
    if 'task_completed' not in st.session_state:
        st.session_state.task_completed = False


def render_sidebar():
    """Rendera sidebar med textval"""
    with st.sidebar:
        st.header("Välj text")
        
        # Toggle mellan färdiga texter och egen text
        text_source = st.radio(
            "Välj textkälla:",
            ["Egen text", "Färdiga texter"],
            horizontal=True
        )
        
        selected_text_info = None
        current_text_key = None
        
        if text_source == "Färdiga texter":
            selected_text_info, current_text_key = handle_predefined_texts()
        elif text_source == "Egen text":
            selected_text_info, current_text_key = handle_custom_text()
        
        # Hantera textbyte
        if current_text_key and st.session_state.current_text_key != current_text_key:
            reset_session_for_new_text(current_text_key, selected_text_info)
        
        # Visa textinfo
        if selected_text_info:
            display_text_info(selected_text_info)
        
        # Rensa chat-knapp
        if st.button("🔄 Ny konversation / Välj ny text", use_container_width=True):
            reset_session_completely()
        
        return selected_text_info


def handle_predefined_texts():
    """Hantera val av färdiga texter"""
    texts = st.session_state.backend.load_available_texts()
    
    if not texts:
        st.error("Inga färdiga texter kunde laddas. Använd 'Egen text' istället.")
        return None, None
    
    text_options = [f"{t['title']} ({t['subject']}, {t['level']})" for t in texts]
    selected_index = st.selectbox(
        "Välj vilken text du vill öva på:",
        range(len(text_options)),
        format_func=lambda x: text_options[x],
        key="text_selector"
    )
    
    selected_text_info = texts[selected_index]
    current_text_key = st.session_state.backend.generate_text_key(selected_text_info, selected_index)
    
    return selected_text_info, current_text_key


def handle_custom_text():
    """Hantera egen text input"""
    st.markdown("**Skriv eller klistra in din egen text:**")
    custom_text = st.text_area(
        "Text att återberätta:",
        height=200,
        placeholder="Klistra in eller skriv texten som du vill öva på att återberätta...",
        key="custom_text_input"
    )
    
    if custom_text.strip():
        selected_text_info = st.session_state.backend.create_custom_text(custom_text)
        current_text_key = st.session_state.backend.generate_text_key(selected_text_info)
        return selected_text_info, current_text_key
    else:
        st.info("Skriv eller klistra in en text för att börja.")
        return None, None


def reset_session_for_new_text(current_text_key, selected_text_info):
    """Återställ session för ny text"""
    st.session_state.current_text_key = current_text_key
    st.session_state.messages = []
    st.session_state.document_data = None
    st.session_state.current_text_info = selected_text_info
    st.session_state.task_completed = False


def reset_session_completely():
    """Återställ hela sessionen"""
    st.session_state.messages = []
    st.session_state.task_completed = False
    st.session_state.current_text_key = None
    st.session_state.document_data = None
    st.session_state.current_text_info = None
    st.rerun()


def display_text_info(selected_text_info):
    """Visa information om vald text"""
    st.markdown(f"""
    <div class="text-info">
    <strong>Titel:</strong> {selected_text_info['title']}<br>
    <strong>Ämne:</strong> {selected_text_info['subject']}<br>
    <strong>Nivå:</strong> {selected_text_info['level']}
    </div>
    """, unsafe_allow_html=True)


def render_main_content(selected_text_info):
    """Rendera huvudinnehåll"""
    if not selected_text_info:
        render_start_page()
        return
    
    # Förbered dokumentdata om det behövs
    prepare_document_data(selected_text_info)
    
    # Skapa flikar
    tab1, tab2 = st.tabs(["💬 Chatta med din coach", "📄 Läs texten"])
    
    with tab1:
        render_chat_tab()
    
    with tab2:
        render_text_tab(selected_text_info)
    

def render_start_page():
    """Rendera startsida"""
    st.info("👈 Välj en text i sidomenyn för att komma igång!")
    
    st.markdown("""
    ### Så här fungerar Återberättelsecoachen:
    
    1. **Välj en text** - Antingen från de färdiga texterna eller lägg in din egen
    2. **Läs texten** - Använd fliken "Läs texten" för att läsa igenom materialet
    3. **Återberätta** - Skriv din återberättelse i chatten
    4. **Få feedback** - Din coach ger dig konstruktiv och uppmuntrande feedback
    5. **Förbättra** - Fortsätt konversationen för att utveckla din återberättelse
    
    ### Tips för bra återberättelser:
    - Läs texten några gånger först
    - Tänk på huvudbudskapet
    - Inkludera viktiga detaljer
    - Använd dina egna ord
    - Fråga om du är osäker på något!
    """)


def prepare_document_data(selected_text_info):
    """Förbered dokumentdata med embeddings"""
    if st.session_state.document_data is None:
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        def progress_callback(current, total):
            progress_text.text(f"Förbereder text... ({current}/{total})")
            progress_bar.progress(current / total)
        
        with st.spinner("Förbereder texten..."):
            embeddings, chunks = st.session_state.backend.process_text_selection(
                selected_text_info, 
                progress_callback
            )
            st.session_state.document_data = {
                'embeddings': embeddings,
                'chunks': chunks
            }
        
        progress_bar.empty()
        progress_text.empty()


def render_chat_tab():
    """Rendera chat-fliken"""
    st.subheader("💬 Chatta med din coach")
    
    # Visa chatthistorik
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Hantera input beroende på om uppgiften är klar
    if st.session_state.task_completed:
        st.info(" 👏 Grattis! Du har klarat den här texten! Klicka på '🔄 Ny konversation' för att öva på en ny text.")
        prompt = st.chat_input(
            "Uppgiften är klar. Klicka på 'Ny konversation' för att fortsätta.",
            disabled=True,
            key="chat_input_disabled"
        )
    else:
        prompt = st.chat_input(
            "Skriv din återberättelse här...",
            disabled=False,
            key="chat_input_active"
        )
    
    # Hantera användarinput
    if prompt:
        handle_user_input(prompt)


def handle_user_input(prompt):
    """Hantera användarens input"""
    # Lägg till användarmeddelande
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generera och visa svar
    with st.chat_message("assistant"):
        with st.spinner("Tänker..."):
            raw_feedback = st.session_state.backend.generate_feedback(
                prompt,
                st.session_state.document_data['embeddings'],
                st.session_state.document_data['chunks'],
                st.session_state.messages[:-1]
            )

        # Kontrollera om uppgiften är klar och ta bort taggen för visning
        if "[UPPGIFT_KLAR]" in raw_feedback:
            st.session_state.task_completed = True
            display_feedback = raw_feedback.replace("[UPPGIFT_KLAR]", "").strip()
        else:
            display_feedback = raw_feedback.strip()

        st.write(display_feedback)  # Bara här!

    # Utvärdering (gömd)
    with st.expander("🔧 Utvecklingsinfo", expanded=False):
        evaluation = st.session_state.backend.evaluator.evaluate_response(prompt, raw_feedback)
        st.write("**Utvärdering:**")
        st.write(evaluation)

    # Lägg till svar i meddelanden
    st.session_state.messages.append({"role": "assistant", "content": display_feedback})


def render_text_tab(selected_text_info):
    """Rendera text-fliken"""
    st.subheader(f"{selected_text_info['title']}")
    
    if selected_text_info['subject'] != 'Egen':
        st.markdown(f"**Ämne:** {selected_text_info['subject']} | **Nivå:** {selected_text_info['level']}")
    
    # Visa texten i en snygg ruta
    st.markdown(
        f"""
        <div class="custom-text-area">
        {selected_text_info['content'].replace(chr(10), '<br>')}
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Statistik om texten
    render_text_statistics(selected_text_info['content'])


def render_text_statistics(text_content):
    """Rendera textstatistik"""
    word_count = len(text_content.split())
    char_count = len(text_content)
    reading_time = max(1, word_count // 100)  # 100 ord per minut
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Antal ord", word_count)
    with col2:
        st.metric("Antal tecken", char_count)
    with col3:
        st.metric("Lästid (min)", reading_time)


def main():
    """Huvudfunktion"""
    # Header
    st.image("robot.jpg", width=100)
    st.markdown('<h1 class="main-header">Återberättelsecoachen</h1>', unsafe_allow_html=True)
    st.markdown("**Välj en text, berätta om den och få personlig feedback!**")
    
    # Initiera session state
    initialize_session_state()
    
    # Rendera sidebar och få vald text
    selected_text_info = render_sidebar()
    
    # Rendera huvudinnehåll
    render_main_content(selected_text_info)




if __name__ == "__main__":
    main()