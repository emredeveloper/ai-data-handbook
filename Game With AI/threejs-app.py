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
        self.response_times = []
        self.strategies_used = {"chase": 0, "flank": 0, "ambush": 0, "predict": 0}
        
        # 🎮 Gelişmiş oyun bağlamı
        self.game_context = """You are an intelligent AI hunter in a 3D arena game. 
Your goal: Track and catch the BLUE CUBE (player).
You are the RED CUBE enemy.

Game Rules:
- Arena: 25x25 units
- You scan every 3 seconds
- Player is fast and unpredictable
- Use strategic thinking to predict movements

Be smart, be strategic, be entertaining!"""
        
        print("\n🤖 AI Vision Agent initialized!")
        print(f"📡 API Key: {api_key[:20]}...{api_key[-10:]}")
        print("🎯 Enhanced game context loaded")
        print("📊 Performance tracking enabled")
    
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
                                    "text": """🎮 GAME ANALYSIS REQUEST

You are the RED CUBE AI enemy. Analyze this screenshot and respond ONLY with valid JSON:

{
  "player_detected": true or false,
  "player_position": {
    "horizontal": "left" or "center" or "right",
    "vertical": "top" or "middle" or "bottom",
    "x_estimate": -12 to 12,
    "z_estimate": -12 to 12
  },
  "distance_estimate": "very_close" or "close" or "medium" or "far",
  "player_velocity": "stationary" or "slow" or "fast" or "unknown",
  "strategy": "chase" or "intercept" or "flank" or "predict" or "search",
  "confidence": 0 to 100,
  "threat_level": "low" or "medium" or "high",
  "next_action": "brief action description",
  "taunt": "fun message to player (optional)"
}

🎯 TARGET: Find the BLUE CUBE (player)
📍 Your position: RED CUBE
⚡ Make it exciting!"""
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
                    
                    # Strateji tracking
                    strategy = analysis.get('strategy', 'search')
                    if strategy in self.strategies_used:
                        self.strategies_used[strategy] += 1
                    
                    print("\n🎯 PARSED ANALYSIS:")
                    print(f"   Player Detected: {analysis.get('player_detected', 'N/A')}")
                    print(f"   Position: {analysis.get('player_position', 'N/A')}")
                    print(f"   Distance: {analysis.get('distance_estimate', 'N/A')}")
                    print(f"   Velocity: {analysis.get('player_velocity', 'N/A')}")
                    print(f"   Confidence: {analysis.get('confidence', 'N/A')}%")
                    print(f"   Strategy: {strategy}")
                    print(f"   Threat Level: {analysis.get('threat_level', 'N/A')}")
                    print(f"   Next Action: {analysis.get('next_action', 'N/A')}")
                    if analysis.get('taunt'):
                        print(f"   💬 AI Says: '{analysis.get('taunt')}'")
                    print(f"\n📈 Stats: {self.successful_detections}/{self.api_call_count} detections")
                    print(f"📊 Strategies: Chase:{self.strategies_used['chase']} | Predict:{self.strategies_used['predict']}")
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

@app.route('/api/ai-taunt', methods=['GET'])
def ai_taunt():
    """AI'dan rastgele mesaj al - Oyunu daha eğlenceli yapar"""
    taunts = [
        "🎯 Seni görüyorum mavi küp!",
        "🏃‍♂️ Kaçamazsın, tahminlerim çok iyi!",
        "🤖 Ben sadece bir AI'yım ama sen çok yavaşsın!",
        "⚡ 3 saniyede bir tarama yapıyorum, hazır ol!",
        "🎮 Bu oyunu kazanacağım, algoritma benden yana!",
        "🔮 Bir sonraki hareketini tahmin ediyorum...",
        "😎 Grok-4 gücü! Görme yeteneğim var!",
        "🎯 Kovalama modu aktif! Saklan bakalım!",
        "🚀 Tahmin algoritması devrede, kaçış yok!",
        "💪 Her taramada daha akıllı oluyorum!"
    ]
    
    import random
    detection_rate = (ai_agent.successful_detections/ai_agent.api_call_count*100) if ai_agent.api_call_count > 0 else 0
    
    # AI mood'a göre mesaj seç
    if detection_rate > 80:
        mood = "confident"
        message_pool = taunts[:5]
    elif detection_rate > 50:
        mood = "hunting"
        message_pool = taunts[3:8]
    else:
        mood = "searching"
        message_pool = taunts[5:]
    
    return jsonify({
        "taunt": random.choice(message_pool),
        "mood": mood,
        "detection_rate": f"{detection_rate:.1f}%",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/game-tips', methods=['GET'])
def game_tips():
    """Oyunculara ipuçları - AI nasıl çalışıyor"""
    tips = [
        {
            "title": "🤖 AI Gerçekten Görüyor",
            "description": "Grok-4 Vision API her 3 saniyede ekran görüntüsü alıp seni analiz ediyor!",
            "tip": "Zigzag hareket et, AI'nın tahminini zorlaştır!"
        },
        {
            "title": "🔮 Tahmin Sistemi",
            "description": "AI son 10 pozisyonunu takip edip gelecekteki yerini tahmin ediyor!",
            "tip": "Ani yön değiştir, tahminleri boşa çıkar!"
        },
        {
            "title": "⚡ 3 Saniye Kuralı",
            "description": "AI 3 saniyede bir tarama yapıyor. Bu sürede strateji değiştirebilirsin!",
            "tip": "Tarama sonrası yön değiştir, AI'yı şaşırt!"
        },
        {
            "title": "🎯 Strateji Değişimleri",
            "description": "AI 4 farklı mod kullanıyor: Search, Chase, Hunt, Predict",
            "tip": "AI'nın modunu takip et, ona göre hareket et!"
        },
        {
            "title": "💎 Kristal Toplama",
            "description": "Kristaller toplayarak puan kazan, ama AI seni görebilir!",
            "tip": "AI uzaktayken kristal topla, yakınken kaç!"
        }
    ]
    
    import random
    return jsonify({
        "tips": tips,
        "random_tip": random.choice(tips),
        "total_tips": len(tips)
    })

@app.route('/api/status', methods=['GET'])
def status():
    """API durumu ve istatistikler"""
    detection_rate = (ai_agent.successful_detections/ai_agent.api_call_count*100) if ai_agent.api_call_count > 0 else 0
    
    return jsonify({
        "status": "running",
        "ai_model": "x-ai/grok-4-fast:free",
        "game_mode": "hunter",
        "scan_interval": "3 seconds",
        "last_analysis": ai_agent.last_analysis,
        "statistics": {
            "total_api_calls": ai_agent.api_call_count,
            "successful_detections": ai_agent.successful_detections,
            "detection_rate": f"{detection_rate:.1f}%",
            "strategies_used": ai_agent.strategies_used,
            "most_used_strategy": max(ai_agent.strategies_used, key=ai_agent.strategies_used.get) if ai_agent.strategies_used else "none"
        },
        "ai_personality": {
            "mood": "hunting" if detection_rate > 70 else "searching",
            "aggression": "high" if detection_rate > 80 else "medium",
            "intelligence": "adaptive"
        },
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
