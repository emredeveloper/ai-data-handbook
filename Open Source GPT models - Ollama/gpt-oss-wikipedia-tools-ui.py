"""
LM Studio Tool Use Demo: Wikipedia Querying Chatbot (Gradio UI)
Bu uygulama, LM Studio uyumlu bir modele (OpenAI API uyumlu) araç çağrılarıyla
Wikipedia'dan içerik çekerek sohbet etmeyi sağlar.
"""

# Standart kütüphaneler
import json
import re
import uuid
import urllib.parse
import urllib.request
from typing import List, Tuple, Optional

# Üçüncü parti
import gradio as gr
from openai import OpenAI


# LM Studio istemcisi
client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
MODEL = "openai/gpt-oss-20b"


def fetch_wikipedia_content(search_query: str) -> dict:
    """Verilen arama sorgusu için Wikipedia içeriğini çeker."""
    try:
        search_url = "https://en.wikipedia.org/w/api.php"

        # En alakalı makaleyi bul
        search_params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": search_query,
            "srlimit": 1,
        }
        url = f"{search_url}?{urllib.parse.urlencode(search_params)}"
        with urllib.request.urlopen(url) as response:
            search_data = json.loads(response.read().decode())

        if not search_data.get("query", {}).get("search"):
            return {
                "status": "error",
                "message": f"No Wikipedia article found for '{search_query}'",
            }

        normalized_title = search_data["query"]["search"][0]["title"]

        # Bulunan başlıkla içeriği çek
        content_params = {
            "action": "query",
            "format": "json",
            "titles": normalized_title,
            "prop": "extracts",
            "exintro": "true",
            "explaintext": "true",
            "redirects": 1,
        }
        url = f"{search_url}?{urllib.parse.urlencode(content_params)}"
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())

        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return {
                "status": "error",
                "message": f"No Wikipedia article found for '{search_query}'",
            }

        page_id = list(pages.keys())[0]
        if page_id == "-1":
            return {
                "status": "error",
                "message": f"No Wikipedia article found for '{search_query}'",
            }

        content = pages[page_id].get("extract", "").strip()
        return {
            "status": "success",
            "content": content,
            "title": pages[page_id].get("title", normalized_title),
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# LM Studio aracı
WIKI_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_wikipedia_content",
        "description": (
            "Search Wikipedia and fetch the introduction of the most relevant article. "
            "Always use this if the user is asking for something that is likely on wikipedia. "
            "If the user has a typo in their search query, correct it before searching."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "search_query": {
                    "type": "string",
                    "description": "Search query for finding the Wikipedia article",
                },
            },
            "required": ["search_query"],
        },
    },
}


def initialize_messages() -> List[dict]:
    return [
        {
            "role": "system",
            "content": (
                "You are an assistant that can retrieve Wikipedia articles. "
                "When asked about a topic, you can retrieve Wikipedia articles "
                "and cite information from them."
            ),
        }
    ]


def extract_inline_tool_args(assistant_text: str, fallback_query: str) -> Optional[str]:
    """Asistan metninde gömülü bir JSON argümanı varsa yakala; yoksa None döndür."""
    if not assistant_text:
        return None
    json_block_match = re.search(
        r"to=functions\\.fetch_wikipedia_content[^{]*({[\s\S]*?})",
        assistant_text,
    ) or re.search(
        r"(\{[^{}]*\"search_query\"\s*:\s*\".*?\"[^{}]*\})",
        assistant_text,
    )
    if not json_block_match:
        return None
    try:
        args = json.loads(json_block_match.group(1))
        return args.get("search_query") or fallback_query
    except Exception:
        return fallback_query


def build_chat_history(messages: List[dict]) -> List[Tuple[str, str]]:
    """gr.Chatbot bileşeni için (user, assistant) çiftleri oluşturur."""
    history: List[Tuple[str, str]] = []
    pending_user: Optional[str] = None
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        # tool veya system mesajlarını atla
        if role == "system" or role == "tool":
            continue
        # tool_calls içeren assistant mesajlarını (content genelde None) atla
        if role == "assistant" and not content:
            continue
        if role == "user":
            pending_user = content or ""
        elif role == "assistant":
            if pending_user is None:
                # Kullanıcı olmadan asistan mesajı; tek taraflı göster
                history.append(("", content or ""))
            else:
                history.append((pending_user, content or ""))
                pending_user = None
    return history


def chat_once(user_input: str, messages: List[dict]):
    """
    Bir tur kullanıcı girdisini işler, gerekiyorsa aracı çağırır ve
    sohbet ile Wikipedia sonuçlarını döndürür.
    """
    if not user_input.strip():
        return build_chat_history(messages), gr.update(visible=False), gr.update(visible=False), messages

    messages.append({"role": "user", "content": user_input})

    wiki_title_md = gr.update(visible=False)
    wiki_content_md = gr.update(visible=False)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=[WIKI_TOOL],
            tool_choice="auto",
        )

        message_obj = response.choices[0].message

        if getattr(message_obj, "tool_calls", None):
            tool_calls = message_obj.tool_calls

            # tool_calls bilgisini konuşmaya ekle
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": tool_call.function,
                        }
                        for tool_call in tool_calls
                    ],
                }
            )

            # Her aracı çalıştır ve sonuçları ekle
            for tool_call in tool_calls:
                args = json.loads(tool_call.function.arguments)
                result = fetch_wikipedia_content(args.get("search_query", user_input))

                if result.get("status") == "success":
                    wiki_title_md = gr.update(
                        value=f"### Wikipedia: {result.get('title', '')}", visible=True
                    )
                    wiki_content_md = gr.update(
                        value=result.get("content", ""), visible=True
                    )
                else:
                    wiki_title_md = gr.update(
                        value=f"### Wikipedia: Hata", visible=True
                    )
                    wiki_content_md = gr.update(
                        value=f"Hata: {result.get('message', 'Bilinmeyen hata')}",
                        visible=True,
                    )

                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(result),
                        "tool_call_id": tool_call.id,
                    }
                )

            # Araçtan sonra normal yanıtı al
            follow_up = client.chat.completions.create(
                model=MODEL,
                messages=messages,
            )
            assistant_text = follow_up.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": assistant_text})

        else:
            # Inline tool call metni var mı diye kontrol (opsiyonel)
            assistant_text = (message_obj.content or "").strip()
            inline_query = extract_inline_tool_args(assistant_text, user_input)
            if inline_query:
                tool_call_id = f"manual_{uuid.uuid4()}"
                messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": "fetch_wikipedia_content",
                                    "arguments": json.dumps({"search_query": inline_query}),
                                },
                            }
                        ],
                    }
                )
                result = fetch_wikipedia_content(inline_query)
                if result.get("status") == "success":
                    wiki_title_md = gr.update(
                        value=f"### Wikipedia: {result.get('title', '')}", visible=True
                    )
                    wiki_content_md = gr.update(
                        value=result.get("content", ""), visible=True
                    )
                else:
                    wiki_title_md = gr.update(value=f"### Wikipedia: Hata", visible=True)
                    wiki_content_md = gr.update(
                        value=f"Hata: {result.get('message', 'Bilinmeyen hata')}",
                        visible=True,
                    )

                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(result),
                        "tool_call_id": tool_call_id,
                    }
                )

                follow_up = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                )
                assistant_text = follow_up.choices[0].message.content or ""
                messages.append({"role": "assistant", "content": assistant_text})
            else:
                # Normal asistan cevabı
                messages.append({"role": "assistant", "content": assistant_text})

        history = build_chat_history(messages)
        return history, wiki_title_md, wiki_content_md, messages

    except Exception as e:
        # Hata durumunu sohbette göster
        messages.append({"role": "assistant", "content": f"Hata: {str(e)}"})
        history = build_chat_history(messages)
        wiki_title_md = gr.update(value=f"### Wikipedia: Hata", visible=True)
        wiki_content_md = gr.update(
            value=(
                "Lütfen LM Studio sunucusunun 127.0.0.1:1234 adresinde çalıştığından,\n"
                f"'{MODEL}' modelinin indirildiğinden ve yüklü olduğundan emin olun.\n\n"
                f"Hata ayrıntısı: {str(e)}"
            ),
            visible=True,
        )
        return history, wiki_title_md, wiki_content_md, messages


def clear_chat():
    messages = initialize_messages()
    return [], gr.update(value="", visible=False), gr.update(value="", visible=False), messages, ""


with gr.Blocks(title="Wikipedia Araçlı Sohbet (LM Studio)") as demo:
    gr.Markdown(
        """
        ### Wikipedia Araçlı Sohbet (LM Studio)
        - Mesaj kutusuna bir konu/soru yazın ve Gönder'e basın.
        - Model gerekirse Wikipedia aracını kullanarak ilgili makalenin özetini getirir.
        - LM Studio sunucusunun `127.0.0.1:1234` üzerinde çalıştığından emin olun. Model: `openai/gpt-oss-20b`.
        """
    )

    with gr.Row():
        chatbot = gr.Chatbot(label="Sohbet", height=450)

    with gr.Row():
        with gr.Column(scale=3):
            user_input = gr.Textbox(
                label="Mesaj",
                placeholder="Örn: 'Alan Turing kimdir?'",
                lines=2,
            )
            with gr.Row():
                submit_btn = gr.Button("Gönder", variant="primary")
                clear_btn = gr.Button("Temizle")
        with gr.Column(scale=4):
            with gr.Accordion("Wikipedia Sonucu", open=True):
                wiki_title = gr.Markdown(visible=False)
                wiki_content = gr.Markdown(visible=False)

    state = gr.State(initialize_messages())

    submit_btn.click(
        fn=chat_once,
        inputs=[user_input, state],
        outputs=[chatbot, wiki_title, wiki_content, state],
    )

    user_input.submit(
        fn=chat_once,
        inputs=[user_input, state],
        outputs=[chatbot, wiki_title, wiki_content, state],
    )

    clear_btn.click(
        fn=clear_chat,
        inputs=None,
        outputs=[chatbot, wiki_title, wiki_content, state, user_input],
    )


if __name__ == "__main__":
    demo.launch()


