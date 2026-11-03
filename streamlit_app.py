"""Streamlit interface for South Korea export strategy analysis."""

import json
import os

import requests
import streamlit as st

RAG_API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")


def _schedule_scroll(target_id: str) -> None:
    """Scroll to a known page location after Streamlit renders the current run."""
    st.session_state.auto_scroll_target = target_id


def _render_scroll_anchor(target_id: str) -> None:
    """Add a stable DOM target for a deferred scroll."""
    st.html(f'<div id="{target_id}"></div>')


def _render_pending_scroll() -> None:
    """Perform the pending smooth scroll once its destination has rendered."""
    target_id = st.session_state.pop("auto_scroll_target", None)
    if target_id is None:
        return

    encoded_target_id = json.dumps(target_id)
    st.html(
        f"""
        <script>
        requestAnimationFrame(() => requestAnimationFrame(() => {{
            document.getElementById({encoded_target_id})
                ?.scrollIntoView({{ behavior: "smooth", block: "start" }});
        }}));
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def _ask_rag(question: str) -> None:
    """Send a selected question to the RAG API and retain its answer for display."""
    st.session_state.rag_error = None
    st.session_state.rag_last_result = None
    _schedule_scroll("rag-answer-result")
    history = st.session_state.get("rag_history", [])
    request_history = [
        {
            "role": message["role"],
            "content": message["content"][:8000],
        }
        for message in history[-20:]
    ]

    try:
        with st.spinner(
            "Retrieving project documents, then checking whether the ReAct agent "
            "needs supporting sources..."
        ):
            response = requests.post(
                f"{RAG_API_URL}/api/chat",
                json={"question": question, "history": request_history},
                timeout=(5, 180),
            )
    except requests.RequestException as exc:
        st.session_state.rag_error = (
            f"Could not connect to the RAG API at {RAG_API_URL}. "
            "Start the FastAPI backend and try again. "
            f"Details: {exc}"
        )
        return

    if not response.ok:
        try:
            error_response = response.json()
        except ValueError:
            error_response = {}
        detail = (
            error_response.get("detail")
            if isinstance(error_response, dict)
            else None
        ) or response.text or response.reason
        st.session_state.rag_error = (
            f"RAG API returned HTTP {response.status_code}: {detail}"
        )
        return

    try:
        result = response.json()
    except ValueError:
        st.session_state.rag_error = "The RAG API returned an invalid response."
        return

    answer = result.get("answer") if isinstance(result, dict) else None
    if not isinstance(answer, str) or not answer.strip():
        st.session_state.rag_error = "The RAG API response did not contain an answer."
        return

    sources = result.get("sources", [])
    if not isinstance(sources, list):
        sources = []
    tool_steps = result.get("tool_steps", [])
    if not isinstance(tool_steps, list):
        tool_steps = []
    st.session_state.rag_last_result = {
        "question": question,
        "answer": answer,
        "sources": sources,
        "tool_steps": tool_steps,
    }
    st.session_state.rag_history = [
        *history,
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ][-20:]


def _render_question_buttons(questions: list[str], key_prefix: str) -> None:
    """Render clickable questions that each request an answer from the RAG API."""
    for question_number, question in enumerate(questions, start=1):
        if st.button(
            f"{question_number}. {question}",
            key=f"{key_prefix}_question_{question_number}",
            use_container_width=True,
        ):
            _ask_rag(question)


def _render_rag_result() -> None:
    """Display the latest RAG answer, sources, or a request error."""
    error = st.session_state.get("rag_error")
    if error:
        _render_scroll_anchor("rag-answer-result")
        st.error(error)

    result = st.session_state.get("rag_last_result")
    if result:
        _render_scroll_anchor("rag-answer-result")
        st.divider()
        st.subheader("Answer")
        st.caption(f"Question: {result['question']}")
        st.markdown(result["answer"])
        tool_steps = [
            step
            for step in result.get("tool_steps", [])
            if isinstance(step, dict)
            and isinstance(step.get("tool"), str)
            and step["tool"]
        ]
        if tool_steps:
            tool_labels = {
                "primary_document_retrieval": "Primary project-document retrieval",
                "retrieve_documents": "Project document search",
                "wikipedia": "Wikipedia",
                "tavily_search": "Supplementary web search",
            }
            with st.expander("ReAct agent actions"):
                for step in tool_steps:
                    tool_label = tool_labels.get(step["tool"], step["tool"])
                    if step.get("status") == "failed":
                        st.warning(f"Supplementary action failed: {tool_label}")
                    else:
                        st.write(f"Completed action: {tool_label}")

        sources = [
            source
            for source in result["sources"]
            if isinstance(source, dict)
            and isinstance(source.get("source"), str)
            and source["source"]
        ]
        if sources:
            with st.expander(f"Sources ({len(sources)})"):
                for source in sources:
                    source_type = source.get("type")
                    label = f" · {source_type}" if isinstance(source_type, str) else ""
                    st.write(f"{source['source']}{label}")


st.set_page_config(
    page_title="Export Strategy Analysis on South Korea",
    page_icon="🇰🇷",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(37, 99, 235, 0.07), transparent 34rem),
            #f4f7fb;
    }

    .block-container {
        max-width: 1180px;
        padding: 2.5rem 2rem 4rem;
    }

    .hero {
        position: relative;
        overflow: hidden;
        margin: 0 0 2rem;
        padding: 2.8rem 2rem 2.6rem;
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 24px;
        background: linear-gradient(120deg, #10243a 0%, #163b60 62%, #1e4d73 100%);
        box-shadow: 0 18px 42px rgba(16, 36, 58, 0.16);
        color: #fff;
        text-align: center;
    }

    .hero::after {
        position: absolute;
        right: -3rem;
        bottom: -7rem;
        width: 18rem;
        height: 18rem;
        border: 2.5rem solid rgba(255, 255, 255, 0.045);
        border-radius: 50%;
        content: "";
        pointer-events: none;
    }

    .hero-accent {
        width: 3.5rem;
        height: 4px;
        margin: 0 auto 1.15rem;
        border-radius: 999px;
        background: linear-gradient(90deg, #e34f5f 0 50%, #78a8ff 50% 100%);
    }

    .hero-eyebrow {
        margin: 0 0 0.65rem;
        color: #c5d8f1;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
    }

    .hero h1 {
        margin: 0;
        color: #fff;
        font-size: clamp(2rem, 4vw, 3.15rem);
        font-weight: 750;
        letter-spacing: -0.035em;
        line-height: 1.12;
    }

    .hero p {
        max-width: 42rem;
        margin: 0.9rem auto 0;
        color: #d7e3f2;
        font-size: 1.05rem;
        line-height: 1.65;
    }

    h1, h2, h3 {
        color: #142b45;
        text-align: center;
        letter-spacing: -0.02em;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: #40536a;
        line-height: 1.7;
    }

    [data-testid="stMarkdownContainer"] ul,
    [data-testid="stMarkdownContainer"] ol {
        max-width: 980px;
        margin: 0.8rem auto 1.5rem;
        padding-left: 1.6rem;
    }

    [data-testid="stMarkdownContainer"] li {
        padding: 0.28rem 0 0.28rem 0.2rem;
    }

    .hero h1 {
        color: #fff;
    }

    .hero p {
        color: #d7e3f2;
    }

    div[data-testid="stButton"] > button {
        min-height: 3.35rem;
        padding: 0.7rem 1rem;
        border: 1px solid #d6e0ed;
        border-radius: 13px;
        background: linear-gradient(145deg, #fff, #f8faff);
        box-shadow: 0 4px 12px rgba(20, 43, 69, 0.055);
        color: #173653;
        font-weight: 650;
        line-height: 1.4;
        transition: border-color 140ms ease, box-shadow 140ms ease,
            transform 140ms ease, background 140ms ease;
    }

    div[data-testid="stButton"] > button p {
        margin: 0;
        color: #173653;
    }

    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        border-color: #507fbd;
        background: #fff;
        box-shadow: 0 9px 20px rgba(20, 43, 69, 0.12);
        color: #102a43;
    }

    div[data-testid="stButton"] > button:hover p,
    div[data-testid="stButton"] > button:focus-visible p {
        color: #102a43;
    }

    div[data-testid="stButton"] > button:focus-visible {
        border-color: #d84c5b;
        box-shadow: 0 0 0 3px rgba(216, 76, 91, 0.2);
        color: #102a43;
    }

    hr {
        border-color: #dce5ef;
    }

    @media (max-width: 640px) {
        .block-container {
            padding: 1.25rem 1rem 2.5rem;
        }

        .hero {
            padding: 2.1rem 1.2rem;
            border-radius: 18px;
        }

        .hero p {
            font-size: 0.97rem;
        }
    }
    </style>
    <section class="hero">
        <div class="hero-accent"></div>
        <p class="hero-eyebrow">Export strategy · South Korea</p>
        <h1>Explore Korea’s Export Journey</h1>
        <p>Evidence-led perspectives on the policies, capabilities and decisions
        that shaped South Korea’s global competitiveness.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

if "selected_analysis_tab" not in st.session_state:
    st.session_state.selected_analysis_tab = None

tab_columns = st.columns(2)
with tab_columns[0]:
    if st.button("PESTEL", key="analysis_tab_pestel", use_container_width=True):
        st.session_state.selected_analysis_tab = "PESTEL"
        _schedule_scroll("pestel-content")
with tab_columns[1]:
    if st.button(
        "Scenario Analysis",
        key="analysis_tab_scenario",
        use_container_width=True,
    ):
        st.session_state.selected_analysis_tab = "Scenario Analysis"
        _schedule_scroll("scenario-content")

if st.session_state.selected_analysis_tab == "PESTEL":
    _render_scroll_anchor("pestel-content")
    st.header("PESTEL Analysis")
    st.write(
        "Assess the Political, Economic, Social, Technological, Environmental, "
        "and Legal factors relevant to exporting to South Korea."
    )
    pestel_sections = [
        ("Political", "Government policy, political stability, and trade relations."),
        ("Economic", "Growth, inflation, exchange rates, and market conditions."),
        ("Social", "Demographics, consumer behavior, and cultural considerations."),
        ("Technological", "Innovation, infrastructure, and technology adoption."),
        ("Environmental", "Climate policy, sustainability, and environmental risks."),
        ("Legal", "Regulations, standards, tariffs, and legal requirements."),
    ]
    political_questions = [
        "What can Sri Lanka learn from Korea’s continuity in export policy across changes of government?",
        "How did Korea shift from import substitution towards export-oriented industrialisation?",
        "How did Korea use export targets and performance monitoring to guide government support?",
        "What lessons does Korea’s heavy and chemical industry drive offer for selecting industries to support?",
        "What risks of targeted industrial support should Sri Lanka learn from Korea’s experience?",
        "How did Korea combine import liberalisation with continued support for exporters?",
        "How did Korea use trade agreements to expand and diversify overseas markets?",
        "How did Korea respond to restrictions on access to essential imported industrial inputs?",
        "What can Sri Lanka learn from Korea’s negotiations to protect access to major export markets?",
        "How did Korea adapt industrial policy as its production capabilities became more advanced?",
    ]
    economic_questions = [
        "What does Korea’s transition from rapid growth to slower expansion suggest for Sri Lanka’s long-term export planning?",
        "What can Sri Lanka learn from Korea’s economic contraction and recovery during the Asian financial crisis?",
        "How did Korea’s experience during the global financial crisis differ from its experience during the Asian financial crisis?",
        "What lessons does Korea’s pandemic contraction and recovery offer for export-sector resilience?",
        "How does Korea’s inflation experience help explain cost pressures faced by exporters?",
        "What does Korea’s experience reveal about the relationship between currency depreciation, import costs and inflation?",
        "Why does stable inflation matter for exporters’ pricing and investment decisions?",
        "What can Sri Lanka learn from Korea’s move from controlled interest rates towards market-based financing?",
        "How can interest-rate increases affect exporters’ borrowing costs and capacity to invest?",
        "What does Korea’s monetary-policy experience reveal about balancing inflation control, economic growth and financial risks?",
    ]
    social_questions = [
        "What can Sri Lanka learn from Korea’s expansion of basic education during its early development?",
        "How did Korea’s rural social changes help build foundations for industrial development?",
        "How did migration from rural areas to cities support Korea’s industrial workforce?",
        "What does Korea’s experience suggest about preparing workers to move into manufacturing and other non-agricultural activities?",
        "What lessons do Korea’s labour disputes offer for managing working conditions in export industries?",
        "What can Sri Lanka learn from Korea’s expansion of social protection alongside industrial development?",
        "How did the Asian financial crisis expose weaknesses in employment security in Korea?",
        "What lessons do differences between Korea’s regular and non-regular workers offer for inclusive export growth?",
        "What do Korea’s ageing population and reliance on foreign workers suggest for long-term workforce planning?",
        "How could Sri Lanka draw lessons from international interest in Korean culture when promoting products and services abroad?",
    ]
    technology_questions = [
        "What can Sri Lanka learn from the electricity and technical-personnel shortages that constrained Korea’s early industries?",
        "How did Korean firms learn through imported machinery, foreign licences and relationships with overseas buyers?",
        "How did Korea connect improvements in production technology and product quality with the search for overseas markets?",
        "What roles did universities, KIST and KAIST play in building Korea’s technological capabilities?",
        "What does Korea’s heavy-industry experience reveal about the skills needed beyond purchasing machinery?",
        "How did Korea move from learning foreign technologies towards developing its own research and innovations?",
        "What can Sri Lanka learn from Korea’s telecommunications and broadband development?",
        "How did Korea’s digital policies support data, AI and practical applications across the economy?",
        "What do Korea’s technology gaps between large firms and SMEs suggest for designing exporter support?",
        "How should Sri Lanka assess Korea’s robotics and AI initiatives without confusing investment announcements with proven productivity gains?",
    ]
    environmental_questions = [
        "What can Sri Lanka learn from Korea’s forest restoration and management of degraded land?",
        "What lessons does Korea’s industrial expansion offer for managing water pollution?",
        "How did Korea strengthen waste management, recycling and producer responsibility?",
        "What can Sri Lanka learn from Korea’s efforts to link environmental improvement with economic development?",
        "What does Korea’s dependence on imported energy and minerals reveal about industrial vulnerability?",
        "How can access to cleaner electricity influence the competitiveness of export industries?",
        "What can Sri Lanka learn from Korea’s emissions trading system and carbon-neutrality objectives?",
        "What lessons does flooding at POSCO’s Pohang steelworks offer for protecting export production from natural hazards?",
        "Why are overseas carbon requirements and emissions information becoming important for exporters?",
        "What opportunities for export diversification are suggested by Korea’s capabilities in cleaner technologies?",
    ]
    legal_questions = [
        "What can Sri Lanka learn from Korea’s development of a stable legal framework for commercial activity?",
        "How did Korea’s foreign-investment laws evolve to support industrial development and investment?",
        "What lessons do Korea’s competition laws offer for managing market dominance and economic concentration?",
        "How did Korea’s WTO commitments shape the forms of support available to exporters?",
        "What can Sri Lanka learn from Korea’s corporate-governance reforms following the Asian financial crisis?",
        "How did Korea’s trade agreements broaden business obligations beyond tariffs?",
        "What lessons do Korea’s labour standards and working-time reforms offer for export-sector development?",
        "What can Sri Lanka learn from Korea’s anti-corruption rules when administering export support?",
        "What do Korea’s workplace-safety laws suggest about accountability in industrial operations?",
        "What lessons do Korea’s data-protection and AI laws offer for developing digitally enabled exports?",
    ]
    if "selected_pestel_section" not in st.session_state:
        st.session_state.selected_pestel_section = None

    for row_start in range(0, len(pestel_sections), 3):
        columns = st.columns(3)
        for column, (title, description) in zip(
            columns, pestel_sections[row_start : row_start + 3]
        ):
            with column:
                if st.button(
                    title,
                    key=f"pestel_{title.lower()}",
                    use_container_width=True,
                ):
                    st.session_state.selected_pestel_section = title
                    _schedule_scroll("pestel-section-content")

    selected_section = st.session_state.selected_pestel_section
    if selected_section is not None:
        description = next(
            description
            for title, description in pestel_sections
            if title == selected_section
        )
        _render_scroll_anchor("pestel-section-content")
        st.subheader(selected_section)
        st.write(description)
        if selected_section == "Political":
            _render_question_buttons(political_questions, "political")
        elif selected_section == "Economic":
            _render_question_buttons(economic_questions, "economic")
        elif selected_section == "Social":
            _render_question_buttons(social_questions, "social")
        elif selected_section == "Technological":
            _render_question_buttons(technology_questions, "technological")
        elif selected_section == "Environmental":
            _render_question_buttons(environmental_questions, "environmental")
        elif selected_section == "Legal":
            _render_question_buttons(legal_questions, "legal")
        else:
            st.info("Questions for this section will appear here.")
        _render_rag_result()
    _render_pending_scroll()

elif st.session_state.selected_analysis_tab == "Scenario Analysis":
    _render_scroll_anchor("scenario-content")
    st.header("Scenario Analysis")
    st.write(
        "Explore plausible future scenarios for South Korea and their implications "
        "for export strategy."
    )
    scenarios = [
        "South Korea 1977 Export Milestone",
        "South Korea 1964 Export Boost",
        "Korea Export Market Diversification 1974–1977",
        "Hyundai US Market Breakthrough 1986",
        "Establishing a Major Shipbuilding Export Industry in South Korea by 1982",
    ]
    scenario_questions = {
        "South Korea 1977 Export Milestone": [
            "How did South Korea combine export incentives, production capacity and buyer access to sustain growth between 1964 and 1977?",
            "How did access to imported inputs, duty drawback and export finance address different barriers to exporting?",
            "What does Korea’s manufacturing expansion reveal about balancing export promotion with investment in production capacity and productivity?",
            "How did the Heavy and Chemical Industry programme broaden Korea’s export capabilities, and what benefits and costs should Sri Lanka consider when supporting selected industries?",
            "How did Korea develop workforce skills alongside industrial investment, and how should export-related training be evaluated?",
            "How did Korea’s export-promotion meetings help turn national targets into action on firms’ practical problems?",
            "How did KOTRA connect market information, product preparation and buyer contacts, and what can EDB learn from this approach?",
            "What do Korea’s roads, ports and remaining infrastructure bottlenecks suggest about prioritising logistics improvements for exporters?",
            "How did changing overseas demand, market concentration and the oil shock affect Korea’s export expansion?",
            "How could EDB apply lessons from the 1977 milestone to support repeat exports while assessing domestic value creation, public costs, financial risks and working conditions?",
        ],
        "South Korea 1964 Export Boost": [
            "What combination of changes supported South Korea’s US$100 million export breakthrough in 1964, and what can Sri Lanka learn from their interaction?",
            "How did manufactured exports contribute to Korea’s export growth, and what does this suggest about identifying new sources of export earnings?",
            "How might Korea’s 1961 foreign-exchange reform have helped firms recognise profitable export opportunities?",
            "What does Korea’s 1964 devaluation reveal about the balance between higher local-currency export receipts and more expensive imported inputs?",
            "How did tax relief and export finance address different constraints faced by Korean exporters?",
            "How did KOTRA’s early overseas offices, buyer matching and product-promotion activities help connect manufacturers with foreign customers?",
            "How did foreign-exchange shortages and declining external assistance influence Korea’s policy direction, and why was that pressure alone insufficient to generate exports?",
            "How did government targets, monitoring and visible export progress reinforce Korea’s export drive?",
            "Why is the timing of reforms important when deciding which policies contributed to Korea’s breakthrough?",
            "Based on this case, how could EDB connect Sri Lanka’s export targets to firm-level opportunities, financing needs, buyer relationships and measurable additional shipments?",
        ],
        "Korea Export Market Diversification 1974–1977": [
            "Why did South Korea seek export markets beyond the United States and Japan during 1974–1977, and what lessons does this offer for managing export concentration?",
            "How did KOTRA’s market intelligence help manufacturers identify overseas opportunities, and how could EDB apply this approach?",
            "What does Korea’s experience suggest about matching products and supplier capabilities to demand before selecting a new export market?",
            "What can Sri Lanka learn from Korea’s 1974 exhibition in Kuwait about connecting targeted promotion with commercial follow-up?",
            "How did buyer introductions and continuing overseas representation perform different roles in developing export relationships?",
            "How did Korea’s general trading companies help manufacturers reach foreign buyers, and what risks should Sri Lanka consider when supporting similar intermediaries?",
            "How did overseas construction projects create export opportunities for Korean suppliers, and how could Sri Lankan firms explore comparable procurement channels?",
            "What role did government coordination and financial support play in expanding Korea’s overseas construction activities?",
            "How should EDB measure diversification success beyond announced contracts and falling shares in traditional markets?",
            "Based on Korea’s experience, how could EDB design a focused programme linking market research, supplier preparation, buyer access, delivery and repeat orders?",
        ],
        "Hyundai US Market Breakthrough 1986": [
            "How did Hyundai combine pricing, product positioning, distribution and production readiness to enter the US market successfully?",
            "How did Hyundai identify an underserved customer group, and what can Sri Lankan exporters learn from this approach?",
            "What does Hyundai’s pricing strategy reveal about the benefits and limitations of using low prices to enter overseas markets?",
            "How did Hyundai prepare its product for US market requirements, and what lessons does this offer for export readiness?",
            "How did Hyundai’s dealer network and phased geographical expansion support market entry?",
            "What role did Hyundai’s earlier experience in Canada play in its approach to entering the US market?",
            "Why was manufacturing capacity prepared before launch important, and how should exporters coordinate production investment with market development?",
            "How did Hyundai combine foreign technology with its own brand development, and what lessons could Sri Lankan manufacturers draw from this?",
            "Why did Hyundai’s strong initial sales not guarantee lasting competitiveness, and what should exporters monitor beyond sales volume?",
            "Based on Hyundai’s experience, how could Sri Lanka’s Export Development Board support firms before market entry and help them sustain success afterward?",
        ],
        "Establishing a Major Shipbuilding Export Industry in South Korea by 1982": [
            "What can Sri Lanka learn from South Korea’s emergence as a major shipbuilding exporter by 1982?",
            "How did Hyundai’s first overseas orders unlock investment financing, and how could Sri Lankan exporters apply this approach?",
            "How did Korea combine foreign technology with practical training to develop export capabilities?",
            "What does Korea’s experience teach Sri Lanka about linking production investment with confirmed buyer demand?",
            "How did domestic demand support Korean shipbuilders during the global downturn, and what lessons could Sri Lanka consider?",
            "How did private entrepreneurial initiative and government support work together to develop Korea’s shipbuilding industry?",
            "What risks did Hyundai accept through discounted pricing and demanding delivery terms, and what can Sri Lankan exporters learn?",
            "Why did rapid export growth not immediately produce strong profitability, and what should Sri Lanka monitor beyond export revenue?",
            "How could Sri Lanka’s EDB connect overseas buyers, technology partners, financial institutions and training providers?",
            "Which lessons from Korea’s shipbuilding experience could inform Sri Lankan export programmes, and which require further feasibility assessment?",
        ],
    }
    if "selected_scenario" not in st.session_state:
        st.session_state.selected_scenario = None

    for row_start in range(0, len(scenarios), 2):
        columns = st.columns(2)
        for column, scenario in zip(columns, scenarios[row_start : row_start + 2]):
            with column:
                if st.button(
                    scenario,
                    key=f"scenario_{row_start}_{scenario}",
                    use_container_width=True,
                ):
                    st.session_state.selected_scenario = scenario
                    _schedule_scroll("scenario-detail-content")

    if st.session_state.selected_scenario is not None:
        _render_scroll_anchor("scenario-detail-content")
        st.subheader(st.session_state.selected_scenario)
        questions = scenario_questions.get(st.session_state.selected_scenario)
        if questions:
            scenario_key = (
                st.session_state.selected_scenario.lower()
                .replace(" ", "_")
                .replace("–", "_")
            )
            _render_question_buttons(questions, scenario_key)
        else:
            st.info("Questions for this scenario will be added here.")
        _render_rag_result()
    _render_pending_scroll()
