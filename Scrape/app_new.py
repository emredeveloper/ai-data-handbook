import requests
import json
from flask import Flask, render_template, request, jsonify
from bs4 import BeautifulSoup
import os
from openai import OpenAI

app = Flask(__name__)

# OpenRouter API Key
OPENROUTER_API_KEY = "sk-or-v1....."

# OpenAI client for OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def scrape_wikipedia_article(url):
    """Wikipedia makalesinden article text'ini çeker."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        article_content = soup.find('div', {'id': 'mw-content-text'})
        
        if not article_content:
            return None
        
        paragraphs = article_content.find_all('p')
        article_text = '\n\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        return article_text
    except Exception as e:
        print(f"Scraping hatası: {e}")
        return None

def call_grok_ai(prompt, article_text):
    """DeepSeek Chat modeli ile AI çağrısı yapar."""
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost:5000",
                "X-Title": "English Learning Platform",
            },
            model="deepseek/deepseek-chat-v3.1:free",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert English teacher who helps Turkish students learn English through reading comprehension. Always return responses in the exact format requested."
                },
                {
                    "role": "user",
                    "content": f"{prompt}\n\nArticle Text:\n{article_text[:3500]}"
                }
            ],
            temperature=0.7
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        print(f"❌ AI API Error: {e}")
        return f"Error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scrape', methods=['POST'])
def scrape():
    data = request.json
    url = data.get('url')
    
    if not url:
        return jsonify({'error': 'URL gerekli'}), 400
    
    article_text = scrape_wikipedia_article(url)
    
    if not article_text:
        return jsonify({'error': 'Makale çekilemedi'}), 400
    
    return jsonify({
        'success': True,
        'article': article_text,
        'word_count': len(article_text.split())
    })

@app.route('/analyze-level', methods=['POST'])
def analyze_level():
    data = request.json
    article_text = data.get('article')
    
    if not article_text:
        return jsonify({'error': 'Makale metni gerekli'}), 400
    
    prompt = """Analyze this English article and provide a STRUCTURED HTML response.

Use this EXACT HTML format with inline styles:

<div style="font-family: 'Segoe UI', sans-serif; line-height: 1.8;">
    <div style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 20px; border-radius: 12px; margin-bottom: 20px;">
        <h2 style="margin: 0 0 10px 0; font-size: 24px;">📊 İngilizce Seviye Analizi</h2>
        <p style="margin: 0; font-size: 18px; opacity: 0.95;">CEFR Standart Değerlendirmesi</p>
    </div>
    
    <div style="background: #e8f4fb; padding: 20px; border-radius: 10px; margin-bottom: 15px; border-left: 5px solid #2196f3; color: #072031;">
        <h3 style="color: #1976d2; margin: 0 0 10px 0; font-size: 20px;">🎯 Seviye: [A1/A2/B1/B2/C1/C2]</h3>
        <p style="margin: 0; color: #333; font-size: 16px;">[Kısa açıklama - 1-2 cümle]</p>
    </div>
    
    <div style="background: #fff7ed; padding: 20px; border-radius: 10px; margin-bottom: 15px; border-left: 5px solid #ff9800; color: #082022;">
        <h3 style="color: #f57c00; margin: 0 0 10px 0; font-size: 18px;">📚 Önemli Kelimeler (5 adet)</h3>
        <ul style="margin: 0; padding-left: 20px; color: #333;">
            <li><strong>word</strong> - Türkçe anlamı</li>
        </ul>
    </div>
    
    <div style="background: #faf5fb; padding: 20px; border-radius: 10px; margin-bottom: 15px; border-left: 5px solid #9c27b0; color: #082022;">
        <h3 style="color: #7b1fa2; margin: 0 0 10px 0; font-size: 18px;">📊 Zorluk Derecesi</h3>
    <div style="background: var(--surface-1); color: #072031; border-radius: 8px; padding: 10px; margin-top: 10px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="flex: 1; height: 10px; background: #e0e0e0; border-radius: 10px; overflow: hidden;">
                    <div style="width: [X]%; height: 100%; background: linear-gradient(90deg, #4caf50, #ff9800, #f44336);"></div>
                </div>
                <strong style="color: #667eea;">[X]/10</strong>
            </div>
        </div>
    </div>
    
    <div style="background: #eef8f0; padding: 20px; border-radius: 10px; border-left: 5px solid #4caf50; color: #072031;">
        <h3 style="color: #388e3c; margin: 0 0 10px 0; font-size: 18px;">💡 Öneriler</h3>
        <p style="margin: 0; color: #333;">[2-3 pratik öneri]</p>
    </div>
</div>

Keep it SHORT and VISUAL! Use actual CEFR criteria."""
    
    analysis = call_grok_ai(prompt, article_text)
    
    return jsonify({'success': True, 'analysis': analysis})

@app.route('/practice', methods=['POST'])
def practice():
    data = request.json
    article_text = data.get('article')
    practice_type = data.get('type', 'vocabulary')
    
    if not article_text:
        return jsonify({'error': 'Makale metni gerekli'}), 400
    
    # Simplified prompts that return clean JSON
    if practice_type == 'vocabulary':
        prompt = """Create 5 vocabulary exercises from the article. Return ONLY valid JSON:

{"exercises": [
  {"type": "multiple_choice", "question": "What does 'prominence' mean in context?", "options": ["importance", "fame", "power", "courage"], "correct": 1, "explanation": "Prominence = şöhret, tanınma"},
  {"type": "fill_blank", "sentence": "He undertook progressive _____ to modernize Turkey.", "options": ["reforms", "changes", "laws", "rules"], "correct": 0, "explanation": "Reforms = reformlar"},
  {"type": "multiple_choice", "question": "Best Turkish translation of 'secular':", "options": ["dini", "laik", "modern", "özgür"], "correct": 1, "explanation": "Secular = laik"},
  {"type": "true_false", "statement": "Atatürk was Turkey's first president (1923-1938).", "options": ["True", "False"], "correct": 0, "explanation": "Doğru! Makalede belirtiliyor."},
  {"type": "multiple_choice", "question": "'Modernized' means:", "options": ["çağdaşlaştırdı", "değiştirdi", "yeniledi", "bozdu"], "correct": 0, "explanation": "Modernized = çağdaşlaştırdı"}
]}"""
    
    elif practice_type == 'grammar':
        prompt = """Create 5 grammar exercises. Return ONLY valid JSON:

{"exercises": [
  {"type": "multiple_choice", "question": "In 'Atatürk WAS a marshal', which tense?", "options": ["Simple Past", "Present Perfect", "Past Continuous", "Past Perfect"], "correct": 0, "explanation": "Simple Past tense"},
  {"type": "multiple_choice", "question": "Identify the passive voice:", "options": ["He introduced reforms", "The sultanate was abolished", "Turkey modernized", "He served"], "correct": 1, "explanation": "Passive: was abolished"},
  {"type": "fill_blank", "sentence": "He _____ as president from 1923 to 1938.", "options": ["serve", "served", "serving", "serves"], "correct": 1, "explanation": "Past tense: served"},
  {"type": "multiple_choice", "question": "Subject in 'His reforms modernized Turkey'?", "options": ["reforms", "His reforms", "Turkey", "modernized"], "correct": 1, "explanation": "Subject: His reforms"},
  {"type": "true_false", "statement": "'He was introducing' = 'He introduced' (same meaning)", "options": ["True", "False"], "correct": 1, "explanation": "Yanlış! Past Continuous vs Simple Past"}
]}"""
    
    elif practice_type == 'comprehension':
        prompt = """Create 5 comprehension questions. Return ONLY valid JSON:

{"exercises": [
  {"type": "multiple_choice", "question": "What was Atatürk's main role?", "options": ["Military leader only", "Founding father and first president", "Revolutionary writer", "Ottoman sultan"], "correct": 1, "explanation": "Founding father and first president"},
  {"type": "multiple_choice", "question": "When did he serve as president?", "options": ["1920-1930", "1923-1938", "1915-1925", "1930-1940"], "correct": 1, "explanation": "1923-1938"},
  {"type": "true_false", "statement": "Atatürk introduced reforms to modernize Turkey.", "options": ["True", "False"], "correct": 0, "explanation": "Doğru! Modernleştirme reformları"},
  {"type": "multiple_choice", "question": "Where did Atatürk gain prominence?", "options": ["Istanbul", "Ankara", "Defence of Gallipoli", "Europe"], "correct": 2, "explanation": "Çanakkale Savunması"},
  {"type": "multiple_choice", "question": "What type of state did he create?", "options": ["Religious", "Secular", "Monarchic", "Federal"], "correct": 1, "explanation": "Laik (secular) devlet"}
]}"""
    
    else:  # quiz
        prompt = """Create 8 mixed quiz questions. Return ONLY valid JSON:

{"exercises": [
  {"type": "multiple_choice", "question": "'Prominence' means:", "options": ["importance", "fame", "power", "beauty"], "correct": 1, "explanation": "Prominence = şöhret"},
  {"type": "fill_blank", "sentence": "Atatürk _____ as Turkey's first president.", "options": ["serve", "served", "serving", "serves"], "correct": 1, "explanation": "Past tense: served"},
  {"type": "true_false", "statement": "Atatürk introduced secular reforms.", "options": ["True", "False"], "correct": 0, "explanation": "Doğru! Laik reformlar"},
  {"type": "multiple_choice", "question": "When did Atatürk die?", "options": ["1930", "1935", "1938", "1940"], "correct": 2, "explanation": "1938"},
  {"type": "multiple_choice", "question": "'Modernize' means:", "options": ["çağdaşlaştırmak", "bozmak", "değiştirmek", "yenilemek"], "correct": 0, "explanation": "Modernize = çağdaşlaştırmak"},
  {"type": "true_false", "statement": "Atatürk was an Ottoman sultan.", "options": ["True", "False"], "correct": 1, "explanation": "Yanlış! Cumhuriyet kurucusu"},
  {"type": "multiple_choice", "question": "Tense: 'He was serving'?", "options": ["Simple Past", "Past Continuous", "Present Perfect", "Future"], "correct": 1, "explanation": "Past Continuous"},
  {"type": "fill_blank", "sentence": "Turkey became a _____ state.", "options": ["secular", "religious", "monarchic", "federal"], "correct": 0, "explanation": "Secular = laik"}
]}"""
    
    exercises_json = call_grok_ai(prompt, article_text)
    
    # Try to parse JSON with multiple strategies
    try:
        # Remove markdown code blocks
        cleaned = exercises_json
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            parts = cleaned.split("```")
            if len(parts) >= 3:
                cleaned = parts[1].strip()
        
        # Find JSON object boundaries
        if cleaned.find('{') >= 0 and cleaned.rfind('}') > 0:
            start = cleaned.find('{')
            end = cleaned.rfind('}') + 1
            cleaned = cleaned[start:end]
        
        # Parse JSON
        exercises_data = json.loads(cleaned)
        
        print(f"✅ Successfully parsed JSON for {practice_type}")
        print(f"Exercises count: {len(exercises_data.get('exercises', []))}")
        
        return jsonify({'success': True, 'data': exercises_data, 'type': practice_type})
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON Parse Error: {e}")
        print(f"Raw response (first 500 chars): {exercises_json[:500]}")
        
        # Fallback: return as formatted text
        return jsonify({
            'success': True, 
            'exercises': f'<div class="result-box"><h3>⚠️ Format Hatası</h3><p>AI yanıtı JSON formatında değil. Ham yanıt:</p><pre style="white-space: pre-wrap; background: #f5f5f5; padding: 15px; border-radius: 8px;">{exercises_json}</pre></div>', 
            'type': practice_type
        })
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/explain', methods=['POST'])
def explain():
    data = request.json
    selected_text = data.get('text')
    
    if not selected_text:
        return jsonify({'error': 'Metin seçilmedi'}), 400
    
    prompt = f"""You are an English teacher. Provide a STRUCTURED response using HTML.

Selected text: "{selected_text}"

Use this EXACT HTML format with inline styles:

<div style="font-family: 'Segoe UI', sans-serif; font-size: 13px; line-height: 1.6;">
    <div style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 10px 12px; border-radius: 8px; margin-bottom: 12px; font-weight: 600;">
        📚 "{selected_text}"
    </div>
    
    <div style="background: var(--surface-1); padding: 10px 12px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid var(--accent2); color: var(--text-dark);">
        <div style="color: var(--accent2); font-weight: 600; margin-bottom: 4px;">💬 Anlamı</div>
        <div style="color: var(--text-dark);">[Kısa Türkçe açıklama]</div>
    </div>
    
    <div style="background: var(--surface-2); padding: 10px 12px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #ff9800; color: var(--text-dark);">
        <div style="color: #ff9800; font-weight: 600; margin-bottom: 4px;">📝 Gramer</div>
        <div style="color: var(--text-dark);">[Gramer yapısı]</div>
    </div>
    
    <div style="background: var(--surface-2); padding: 10px 12px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #4caf50; color: var(--text-dark);">
        <div style="color: #4caf50; font-weight: 600; margin-bottom: 4px;">🔤 Kelimeler</div>
        <div style="color: var(--text-dark);">
            • <strong>word1</strong>: anlam<br>
            • <strong>word2</strong>: anlam
        </div>
    </div>
    
    <div style="background: var(--surface-2); padding: 10px 12px; border-radius: 8px; border-left: 4px solid #9c27b0; color: var(--text-dark);">
        <div style="color: #9c27b0; font-weight: 600; margin-bottom: 4px;">💡 Örnek</div>
        <div style="color: var(--text-muted); font-style: italic;">[İngilizce örnek cümle]</div>
    </div>
</div>

Keep it SHORT - max 2 sentences per section!"""
    
    explanation = call_grok_ai(prompt, selected_text)
    
    return jsonify({'success': True, 'explanation': explanation})


@app.route('/vocab-by-level', methods=['POST'])
def vocab_by_level():
    data = request.json
    article_text = data.get('article')
    level = data.get('level', 'A1').upper()

    if not article_text:
        return jsonify({'error': 'Makale metni gerekli'}), 400

    prompt = f"""You are an English teacher and lexicographer. Given this article and a CEFR level ({level}), select 10 vocabulary words from the article that are appropriate for that level. For each word, return a JSON object with:
    - word
    - part_of_speech (noun/verb/adjective/adverb)
    - short_turkish (one-line Turkish translation)
    - example (one clear English example sentence using the word in the article context)

Return valid JSON: {{"words": [{{...}}]}}. Keep each example short (max 12 words). Use the article context when possible."""

    ai_response = call_grok_ai(prompt, article_text)

    # try to extract JSON
    cleaned = ai_response
    try:
        if '```json' in cleaned:
            cleaned = cleaned.split('```json')[1].split('```')[0].strip()
        elif '```' in cleaned:
            parts = cleaned.split('```')
            if len(parts) >= 3:
                cleaned = parts[1].strip()

        if cleaned.find('{') >= 0 and cleaned.rfind('}') > 0:
            start = cleaned.find('{')
            end = cleaned.rfind('}') + 1
            cleaned = cleaned[start:end]

        data_out = json.loads(cleaned)
        return jsonify({'success': True, 'data': data_out, 'level': level})
    except Exception as e:
        print('Vocab-by-level parse error:', e)
        return jsonify({'success': False, 'error': 'AI yanıtı parse edilemedi', 'raw': ai_response})

if __name__ == '__main__':
    print("🚀 English Learning Platform başlatılıyor...")
    print("📚 Wikipedia makalesiyle İngilizce öğrenin!")
    print("🌐 http://localhost:5000")
    app.run(debug=True, port=5000)
