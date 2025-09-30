"""
AI Vision Game Backend - OpenRouter Grok-4 ile Görsel Oyun
AI gerçekten görebilir ve oyuncuyu takip eder!
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import requests
import json
import base64
from io import BytesIO
from PIL import Image
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# OpenRouter API anahtarınızı buraya ekleyin
OPENROUTER_API_KEY = "sk-or-v1...."  # Kendi API anahtarınızı buraya yazın
SITE_URL = "http://localhost:5000"
SITE_NAME = "AI Vision Game"

class AIVisionAgent:
    """Görsel AI Ajanı - Oyun ekranını analiz eder"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.last_analysis = None
        self.api_call_count = 0
        self.successful_detections = 0
        self.game_context = "You are an AI enemy in a 3D game. Analyze the game screenshot and find the player (blue cube)."
        print("\n🤖 AI Vision Agent initialized!")
        print(f"📡 API Key: {api_key[:20]}...{api_key[-10:]}")
    
    def analyze_game_screen(self, image_base64):
        """
        Oyun ekranını AI ile analiz et
        AI'ın oyuncuyu bulmasını sağla
        """
        self.api_call_count += 1
        print(f"\n{'='*60}")
        print(f"🔍 AI SCAN #{self.api_call_count} BAŞLADI")
        print(f"📸 Image size: {len(image_base64)} bytes")
        print(f"🤖 Model: x-ai/grok-4-fast:free")
        print(f"{'='*60}")
        
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": SITE_URL,
                    "X-Title": SITE_NAME,
                },
                data=json.dumps({
                    "model": "x-ai/grok-4-fast:free",
                    "messages": [
                        {
                            "role": "system",
                            "content": self.game_context
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": """Analyze this game screenshot and answer in JSON format:
{
  "player_detected": true/false,
  "player_location": "left/center/right/top/bottom",
  "distance_estimate": "close/medium/far",
  "strategy": "chase/flank/ambush",
  "confidence": 0-100,
  "reasoning": "brief explanation"
}

The blue cube is the player. Find it and track its position."""
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    "temperature": 0.7,
                    "max_tokens": 500
                })
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['choices'][0]['message']['content']
                
                print("✅ API Response received!")
                print(f"📊 Model used: {result.get('model', 'unknown')}")
                print(f"💬 Raw AI Response:\n{ai_response[:300]}..." if len(ai_response) > 300 else f"💬 Raw AI Response:\n{ai_response}")
                
                # JSON'u parse et
                try:
                    # Markdown kod bloğu içindeyse temizle
                    if '```json' in ai_response:
                        ai_response = ai_response.split('```json')[1].split('```')[0].strip()
                    elif '```' in ai_response:
                        ai_response = ai_response.split('```')[1].split('```')[0].strip()
                    
                    analysis = json.loads(ai_response)
                    self.last_analysis = analysis
                    
                    # Başarılı tespit sayacı
                    if analysis.get('player_detected'):
                        self.successful_detections += 1
                    
                    print("\n🎯 PARSED ANALYSIS:")
                    print(f"   Player Detected: {analysis.get('player_detected', 'N/A')}")
                    print(f"   Location: {analysis.get('player_location', 'N/A')}")
                    print(f"   Confidence: {analysis.get('confidence', 'N/A')}%")
                    print(f"   Strategy: {analysis.get('strategy', 'N/A')}")
                    print(f"   Reasoning: {analysis.get('reasoning', 'N/A')}")
                    print(f"\n📈 Stats: {self.successful_detections}/{self.api_call_count} successful detections")
                    print(f"{'='*60}\n")
                    
                    return analysis
                except json.JSONDecodeError as e:
                    # Eğer parse edilemezse, varsayılan yanıt dön
                    print(f"❌ JSON Parse Error: {str(e)}")
                    print(f"📄 Raw response that failed: {ai_response}")
                    print(f"{'='*60}\n")
                    return {
                        "player_detected": False,
                        "player_location": "unknown",
                        "distance_estimate": "far",
                        "strategy": "search",
                        "confidence": 0,
                        "reasoning": "Failed to parse AI response",
                        "raw_response": ai_response
                    }
            else:
                print(f"❌ API Error: Status {response.status_code}")
                print(f"📄 Response: {response.text}")
                print(f"{'='*60}\n")
                return {
                    "error": f"API Error: {response.status_code}",
                    "player_detected": False,
                    "strategy": "random"
                }
                
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR in AI Analysis: {str(e)}")
            import traceback
            print(f"📋 Traceback:\n{traceback.format_exc()}")
            print(f"{'='*60}\n")
            return {
                "error": str(e),
                "player_detected": False,
                "strategy": "random"
            }
    
    def get_simple_analysis(self, image_base64):
        """
        Daha basit analiz - oyuncu pozisyonu için
        """
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": SITE_URL,
                    "X-Title": SITE_NAME,
                },
                data=json.dumps({
                    "model": "x-ai/grok-4-fast:free",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Where is the BLUE cube in this image? Answer only: 'left', 'right', 'center', 'top', 'bottom', or 'not visible'"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    "temperature": 0.3,
                    "max_tokens": 50
                })
            )
            
            if response.status_code == 200:
                result = response.json()
                location = result['choices'][0]['message']['content'].strip().lower()
                return {"location": location, "detected": "not" not in location}
            
            return {"location": "unknown", "detected": False}
            
        except Exception as e:
            print(f"Simple Analysis Error: {str(e)}")
            return {"location": "unknown", "detected": False}


# AI Ajanı oluştur
ai_agent = AIVisionAgent(OPENROUTER_API_KEY)

@app.route('/')
def index():
    """Ana sayfa - Oyun HTML'i"""
    return render_template('game_ai_vision.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_screen():
    """Oyun ekranını AI ile analiz et"""
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({"error": "No image provided"}), 400
        
        # Base64'ü temizle (data:image/png;base64, kısmını kaldır)
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        # AI analizi
        analysis = ai_agent.analyze_game_screen(image_data)
        
        return jsonify({
            "success": True,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/simple-analyze', methods=['POST'])
def simple_analyze():
    """Basit pozisyon analizi"""
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({"error": "No image provided"}), 400
        
        # Base64'ü temizle
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        # Basit analiz
        result = ai_agent.get_simple_analysis(image_data)
        
        return jsonify({
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/status', methods=['GET'])
def status():
    """API durumu"""
    return jsonify({
        "status": "running",
        "ai_model": "x-ai/grok-4-fast:free",
        "last_analysis": ai_agent.last_analysis,
        "total_api_calls": ai_agent.api_call_count,
        "successful_detections": ai_agent.successful_detections,
        "detection_rate": f"{(ai_agent.successful_detections/ai_agent.api_call_count*100):.1f}%" if ai_agent.api_call_count > 0 else "0%",
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    # templates klasörü oluştur
    os.makedirs('templates', exist_ok=True)
    
    print("=" * 60)
    print("🎮 AI VISION GAME - OpenRouter Grok-4 ile Oyun")
    print("=" * 60)
    print(f"🤖 AI Model: x-ai/grok-4-fast:free")
    print(f"🌐 Server: http://localhost:5000")
    print(f"📸 AI oyun ekranını gerçekten görebilir ve analiz eder!")
    print("=" * 60)
    print("\n⚠️  ÖNEMLİ: OPENROUTER_API_KEY değişkenine API anahtarınızı ekleyin!")
    print("🔑 API Key: https://openrouter.ai/keys\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
