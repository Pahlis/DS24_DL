"""
Återberättelsecoachen - En AI-driven applikation för att träna återberättelse
================================================================

Detta är en Streamlit-applikation som hjälper användare att träna sina färdigheter
inom återberättelse genom att använda AI (Google Gemini) för att ge feedback.

Applikationen låter användare:
1. Välja mellan färdiga texter eller ladda upp egna texter
2. Läsa texten i ett användarvänligt gränssnitt
3. Återberätta texten i en chatbot-miljö
4. Få konstruktiv feedback från AI:n
5. Fortsätta dialogen för att förbättra återberättelsen

Teknisk stack:
- Streamlit för webbgränssnittet
- Google Gemini API för AI-feedback
- Embeddings för textanalys och relevansmatching
- Session state för att hantera användardata mellan interaktioner

Författare: Lisa Påhlsson
Datum: 2025-05-30
Kurs: Deep Learning
"""

import streamlit as st
import sys
import os

try:
    from backend import AppBackend
except ImportError as e:
    #Felhantering om backend.py inte kan importeras
    st.error(f"Kunde inte importera backend: {e}")
    st.error("Se till att backend.py finns i samma mapp som app.py")
    st.stop()

# Konfiguration av Streamlit-sidan
st.set_page_config(
    page_title="Återberättelsecoachen",
    layout="wide"
)

# CSS för anpassat utseende
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
    """
    Initialiserar alla session state-variabler som behövs för applikationen.
    Detta säkerställer att alla nödvändiga data är tillgängliga
    """
    if "backend" not in st.session_state:
        try:
            # Hämtar API-nyckel från Streamlit-secrets
            api_key = st.secrets["GEMINI_API_KEY"]
            st.session_state.backend = AppBackend(api_key)
        except KeyError:
            # Om API-nyckeln inte finns, visa felmeddelande och appen stoppar
            st.error("Fel: GEMINI_API_KEY kunde inte hittas. Se till att den är satt i .streamlit/secrets.toml lokalt eller som secret i Streamlit Cloud.")
            st.stop()
        except Exception:
            # Om något annat fel uppstod vid initialisering för att säkerställa att appen inte kraschar eller API exponeras
            st.error("Ett oväntat fel uppstod vid initialisering av backend. Kontrollera din konfiguration.")
            st.stop()
    
    # Initiera chatthistorik som tom lista
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Spårar vilken text som är vald för att hantera textbyten
    if "current_text_key" not in st.session_state:
        st.session_state.current_text_key = None
    
    # Lagrar processad textdata (embeddings och chunks)
    if "document_data" not in st.session_state:
        st.session_state.document_data = None
    
    # Information om den aktuella texten (titel, innehåll, etc.)
    if "current_text_info" not in st.session_state:
        st.session_state.current_text_info = None
    
    # Flagga för att spåra om användaren slutfört uppgiften
    if "task_completed" not in st.session_state:
        st.session_state.task_completed = False


def render_sidebar():
    """
    Skapar sidomenyn där användare kan välja texter och hantera sessionen.    
    Funktionalitet inkluderar:
    - Val mellan färdiga texter och egen text
    - Textval och texthantering
    - Återställning av session
    - Visning av textinformation    
    """
    with st.sidebar:
        st.header("Välj text")
        
        # Välja mellan egen text eller färdiga texter med radio-knappar
        text_source = st.radio(
            "Välj textkälla:",
            ["Egen text", "Färdiga texter"],
            horizontal=True
        )
        
        # Hanterar val av textkälla
        selected_text_info = None
        current_text_key = None
        
        # Hanterar olika textkällor med separata funktioner
        if text_source == "Färdiga texter":
            selected_text_info, current_text_key = handle_predefined_texts()
        elif text_source == "Egen text":
            selected_text_info, current_text_key = handle_custom_text()
        
        # Hanterar textbyte
        if current_text_key and st.session_state.current_text_key != current_text_key:
            reset_session_for_new_text(current_text_key, selected_text_info)
        
        # Visar textinfo
        if selected_text_info:
            display_text_info(selected_text_info)
        
        # Rensar chat-knapp (inkl ikon för ökad tydlighet)
        if st.button("🔄 Ny konversation / Välj ny text", use_container_width=True):
            reset_session_completely()
        
        return selected_text_info


def handle_predefined_texts():
    """
    Hanterar val av färdigdefinierade texter från backend.    
    Laddar tillgängliga texter från backend och låter användaren välja
    via en dropdown-meny. Varje text har titel, ämne och svårighetsgrad.    
    """
    # Laddar tillgängliga texter från backend
    texts = st.session_state.backend.load_available_texts()
    
    if not texts:
        st.error("Inga färdiga texter kunde laddas. Använd 'Egen text' istället.")
        return None, None
    
    #Dropdown för att välja text
    text_options = [f"{t['title']} ({t['subject']}, {t['level']})" for t in texts]
    selected_index = st.selectbox(
        "Välj vilken text du vill öva på:",
        range(len(text_options)),
        format_func=lambda x: text_options[x],
        key="text_selector"
    )
    
    # Hämta vald textinformation och generera nyckel
    selected_text_info = texts[selected_index]
    current_text_key = st.session_state.backend.generate_text_key(selected_text_info, selected_index)
    
    return selected_text_info, current_text_key


def handle_custom_text():
    """
    Hanterar input av egen text från användaren. Låter användaren skriva eller klistra in egen text som de vill träna
    återberättelse på. Använder dynamiska nycklar för att kunna återställa
    textfältet när det behövs.
    """
    
    # Dynamisk nyckel för att kunna återställa textfältet
    if "text_key" not in st.session_state:
        st.session_state.text_key = 0
    
    st.markdown("**Skriv eller klistra in din egen text:**")
    
    # Textområde för att mata in egen text
    custom_text = st.text_area(
        "Text att återberätta:",
        height=200,
        placeholder="Klistra in eller skriv texten som du vill öva på att återberätta...",
        key=f"custom_text_input_{st.session_state.text_key}"  # Dynamisk nyckel
    )

    # knapp för att bekräfta och registrera texten
    if st.button("Bekräfta text"):
        if custom_text.strip():
            selected_text_info = st.session_state.backend.create_custom_text(custom_text)
            current_text_key = st.session_state.backend.generate_text_key(selected_text_info)
            st.success("Texten är registrerad!")
            return selected_text_info, current_text_key
        else:
            st.info("Skriv eller klistra in en text för att börja.")
            return None, None
    return None, None


def reset_session_for_new_text(current_text_key, selected_text_info):
    """
    Återställer session-tillstånd när en ny text väljs.    
    Detta för att säkerställa att chatthistorik och annan data
    från tidigare texter inte blandas ihop med den nya texten.
    """
    st.session_state.current_text_key = current_text_key
    st.session_state.messages = [] #Tömmer chatthistoriken
    st.session_state.document_data = None #Nollställer dokumentdata
    st.session_state.current_text_info = selected_text_info
    st.session_state.task_completed = False #Återställer uppgiftsstatus


def reset_session_completely():
    """
    Utför en fullständig återställning av hela sessionen.
    Används när användaren vill börja helt om från början.
    Detta rensar all chatthistorik, textdata och återställer
    """
    st.session_state.messages = []
    st.session_state.task_completed = False
    st.session_state.current_text_key = None
    st.session_state.document_data = None
    st.session_state.current_text_info = None
    st.session_state.text_key += 1 #Tvingar texten att återställas
    st.rerun()


def display_text_info(selected_text_info):
    """
    Visar information om den valda texten i sidomenyn.    
    Presenterar titel, ämne och svårighetsgrad med HTML-formatering.
    """
    st.markdown(f"""
    <div class="text-info">
    <strong>Titel:</strong> {selected_text_info["title"]}<br>
    <strong>Ämne:</strong> {selected_text_info["subject"]}<br>
    <strong>Nivå:</strong> {selected_text_info["level"]}
    </div>
    """, unsafe_allow_html=True)


def render_main_content(selected_text_info):
    """
    Skapar huvudinnehållet i applikationen.
    Visar antingen startsidan (om ingen text är vald) eller
    huvudgränssnittet med chat och text-flikar.
    """
    if not selected_text_info:
        render_start_page()
        return
    
    # Förbereder dokumentdata om det behövs
    prepare_document_data(selected_text_info)
    
    #Skapar flikar för chat och text med ikoner för ökad tydlighet
    tab1, tab2 = st.tabs(["💬 Chatta med din coach", "📄 Läs texten"])
    
    with tab1:
        render_chat_tab()
    
    with tab2:
        render_text_tab(selected_text_info)
    

def render_start_page():
    """
    Skapar startsidan som visas när ingen text är vald.    
    Inkluderar instruktioner och tips för hur applikationen fungerar.
    """
    st.info("👈 Välj en text i sidomenyn för att komma igång!")

# Expanderbar sektion med detaljerade instruktioner
with st.expander("Instruktioner", expanded=True):
    st.markdown("""
    ### Så här fungerar Återberättelsecoachen:

    1. **Välj en text** - Antingen från de färdiga texterna eller lägg in din egen
    2. **Läs texten** - Använd fliken "Läs texten" för att läsa igenom materialet
    3. **Återberätta** - Skriv din återberättelse i chatten
    4. **Få feedback** - Din coach ger dig konstruktiv och uppmuntrande feedback
    5. **Förbättra** - Fortsätt konversationen för att utveckla din återberättelse
    """)

    st.markdown("""
    <div style="background-color:#f9f9f9;padding:10px;border-radius:5px;border-left:5px solid #1f77b4;">
    <strong>Tips för bra återberättelser:</strong><br>
    ✔ Läs texten några gånger först<br>
    ✔ Tänk på huvudbudskapet<br>
    ✔ Inkludera viktiga detaljer<br>
    ✔ Använd dina egna ord<br>
    ✔ Fråga om du är osäker på något!
    </div>
""", unsafe_allow_html=True)


def prepare_document_data(selected_text_info):
    """
    Förbereder och processar textdata med embeddings för AI-analys.    
    Den skapar vektorrepresentationer (embeddings) av texten som
    möjliggör semantisk analys och relevansmatching.
    """
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
    """
    Skapar chat-fliken där användaren interagerar med AI-coachen.    
    Hanterar:
    - Visning av chatthistorik
    - Användarinput
    - Blockering av input när uppgiften är klar
    - Meddelanden baserat på uppgiftsstatus
    """

    st.subheader("💬 Chatta med din coach")
    
    # Visa chatthistorik
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Hantera input beroende på om uppgiften är klar
    if st.session_state.task_completed:
        st.info(" 👏 Grattis! Du har klarat den här texten! Vilken stjärna du är på att återberätta. Klicka på '🔄 Ny konversation' för att öva på en ny text.")
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
    """
    Hanterar och bearbetar användarens input i chatten.
    Tar emot användarens återberättelse, lägger till den i chatthistoriken
    och genererar AI:s feedback baserat på den.
    """

    # Lägger till användarmeddelande och visar det i chatten
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Genererar och visar svar
    with st.chat_message("assistant"):
        with st.spinner("Tänker..."):
            raw_feedback = st.session_state.backend.generate_feedback(
                prompt,
                st.session_state.document_data['embeddings'],
                st.session_state.document_data['chunks'],
                st.session_state.messages[:-1]
            )

        # Kontrollerar om uppgiften är klar och ta bort taggen för visning
        if "[UPPGIFT_KLAR]" in raw_feedback:
            st.session_state.task_completed = True
            display_feedback = raw_feedback.replace("[UPPGIFT_KLAR]", "").strip()
        else:
            display_feedback = raw_feedback.strip()

        # Visar feedback från AI i en snygg ruta
        st.write(display_feedback)  

    # Utvärdering och utvecklingsinfo i expanderbar sektion
    with st.expander("🔧 Utvecklingsinfo", expanded=False):
        evaluation = st.session_state.backend.evaluator.evaluate_response(prompt, raw_feedback)
        st.write("**Utvärdering:**")
        st.write(evaluation)

    # Lägg till feedback i session state
    st.session_state.messages.append({"role": "assistant", "content": display_feedback})


def render_text_tab(selected_text_info):
    """
    Skapar fliken där användaren kan läsa den valda texten.
    Visar textens titel, ämne och innehåll i en stilad ruta.
    Inkluderar även statistik om texten som ordantal, teckenantal och lästid.
    """
    st.subheader(f"{selected_text_info['title']}")
    
    if selected_text_info["subject"] != "Egen":
        st.markdown(f"**Ämne:** {selected_text_info['subject']} | **Nivå:** {selected_text_info['level']}")
    
    # Visa texten i en stylad ruta
    st.markdown(
        f"""
        <div class="custom-text-area">
        {selected_text_info["content"].replace(chr(10), "<br>")}
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Statistik om texten
    render_text_statistics(selected_text_info["content"])


def render_text_statistics(text_content):
    """
    Beräknar och visar statistik om den valda texten.
    
    Statistiken hjälper användaren att förstå textens komplexitet
    och ungefärlig lästid. Lästiden baseras på genomsnittlig läshastighet
    för barn (ca 100 ord per minut).
    """
    word_count = len(text_content.split())
    char_count = len(text_content)
    reading_time = max(1, word_count // 100)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Antal ord", word_count)
    with col2:
        st.metric("Antal tecken", char_count)
    with col3:
        st.metric("Lästid (min)", reading_time)


def main():
    """
    Huvudfunktion för applikationen.
    Initierar session state, skapar sidomenyn och huvudinnehållet.
    """
    # Rubrik och introduktion
    st.markdown('<h1 class="main-header">Återberättelsecoachen</h1>', unsafe_allow_html=True)
    st.markdown("**Välj en text, berätta om den och få personlig feedback!**")
    
    # Initiera session state
    initialize_session_state()
    
    # Skapar sidebar och få vald text
    selected_text_info = render_sidebar()
    
    # Skapar huvudinnehåll
    render_main_content(selected_text_info)


# Kör appen
if __name__ == "__main__":
    main()


