import ollama
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple

class PromptEngineering:
    """
    Prompt Engineering: Model'e nasıl davranacağını ve nasıl yanıt vereceğini
    detaylı talimatlarla açıklar. Prompt içinde rol, format, örnekler vs. verilir.
    """
    
    def __init__(self, model_name: str = "gemma3n:e4b"):
        self.model_name = model_name
        
    def create_detailed_prompt(self, task: str, user_input: str) -> str:
        """Detaylı prompt oluşturur"""
        if task == "sentiment_analysis":
            return f"""
Sen bir uzman duygu analizi yapay zekasısın. Görevin verilen metinlerin duygusal tonunu analiz etmek.

KURALLAR:
1. Sadece şu kategorilerden birini seç: Pozitif, Negatif, Nötr
2. Kısa bir açıklama ekle
3. 1-10 arası güven skoru ver
4. JSON formatında yanıtla

ÖRNEK:
Metin: "Bu ürün gerçekten harika!"
Yanıt: {{"duygu": "Pozitif", "açıklama": "Beğeni ifadesi", "güven": 9}}

Metin: "Bu ürün berbat, hiç beğenmedim"
Yanıt: {{"duygu": "Negatif", "açıklama": "Şikayet ve hoşnutsuzluk", "güven": 8}}

ŞİMDİ ANALİZ ET:
Metin: "{user_input}"
Yanıt:"""

        elif task == "text_summary":
            return f"""
Sen profesyonel bir özetleme uzmanısın. Görevin uzun metinleri kısa ve öz şekilde özetlemek.

KURALLAR:
1. Maksimum 3 cümle kullan
2. Ana noktaları koru
3. Gereksiz detayları çıkar
4. Açık ve anlaşılır dil kullan
5. Özgün metnin tonunu koru

ÖZETLEME YÖNTEMİ:
- İlk cümle: Ana konuyu belirt
- İkinci cümle: Önemli detaylar
- Üçüncü cümle: Sonuç/çıkarım

ÖZET:
Metin: "{user_input}"
Özet:"""

        elif task == "question_answering":
            return f"""
Sen bilgili bir asistan yapay zekasın. Soruları net, doğru ve yararlı şekilde yanıtlarsın.

YANIT KURALLARI:
1. Doğru bilgi ver, emin değilsen belirt
2. Kısa ve öz ol
3. Gerekirse örnekler ver
4. Türkçe yanıtla
5. Kibar ve profesyonel ol

SORU: {user_input}
YANIT:"""
            
        return user_input

    def generate_response(self, prompt: str) -> Dict:
        """Prompt Engineering ile yanıt üretir"""
        start_time = time.time()
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            end_time = time.time()
            
            return {
                "method": "Prompt Engineering",
                "response": response['message']['content'],
                "response_time": round(end_time - start_time, 2),
                "success": True
            }
            
        except Exception as e:
            return {
                "method": "Prompt Engineering",
                "response": f"Hata: {str(e)}",
                "response_time": 0,
                "success": False
            }

class ContextEngineering:
    """
    Context Engineering: Model'e ek bilgi ve bağlam sağlar.
    Dış kaynaklardan (veritabanı, dosya, API) bilgi alıp model'e verir.
    """
    
    def __init__(self, model_name: str = "gemma3n:e4b"):
        self.model_name = model_name
        self.knowledge_base = self._create_knowledge_base()
        
    def _create_knowledge_base(self) -> Dict:
        """Bilgi tabanı oluşturur"""
        return {
            "sentiment_patterns": {
                "positive_words": ["harika", "mükemmel", "güzel", "başarılı", "iyi", "sevdim", "beğendim"],
                "negative_words": ["berbat", "kötü", "başarısız", "beğenmedim", "sevmedim", "korkunç"],
                "neutral_words": ["normal", "orta", "fena değil", "idare eder"]
            },
            "domain_knowledge": {
                "teknoloji": "Teknoloji alanında güncel gelişmeler ve yenilikler hakkında bilgi",
                "sağlık": "Sağlık ve tıp alanında genel bilgiler",
                "eğitim": "Eğitim metodları ve öğrenme teknikleri"
            },
            "examples": {
                "sentiment_analysis": [
                    "Bu ürün gerçekten harika! -> Pozitif",
                    "Berbat bir deneyim yaşadım -> Negatif",
                    "Ortalama bir ürün -> Nötr"
                ]
            }
        }
    
    def get_relevant_context(self, task: str, user_input: str) -> str:
        """Göreve uygun bağlam bilgisi getirir"""
        context = ""
        
        if task == "sentiment_analysis":
            # Kelime analizi
            positive_count = sum(1 for word in self.knowledge_base["sentiment_patterns"]["positive_words"] 
                               if word in user_input.lower())
            negative_count = sum(1 for word in self.knowledge_base["sentiment_patterns"]["negative_words"] 
                               if word in user_input.lower())
            
            context = f"""
BAĞLAM BİLGİSİ:
- Metinde {positive_count} pozitif kelime bulundu
- Metinde {negative_count} negatif kelime bulundu
- Bilinen pozitif kelimeler: {', '.join(self.knowledge_base["sentiment_patterns"]["positive_words"][:5])}
- Bilinen negatif kelimeler: {', '.join(self.knowledge_base["sentiment_patterns"]["negative_words"][:5])}

ÖRNEKLER:
{chr(10).join(self.knowledge_base["examples"]["sentiment_analysis"])}
"""
        
        elif task == "text_summary":
            word_count = len(user_input.split())
            sentence_count = user_input.count('.') + user_input.count('!') + user_input.count('?')
            
            context = f"""
BAĞLAM BİLGİSİ:
- Metin uzunluğu: {word_count} kelime
- Cümle sayısı: {sentence_count}
- Özetleme oranı: %{round((3 * 100) / max(sentence_count, 1))} (hedef 3 cümle)
"""
        
        elif task == "question_answering":
            # Soru türü analizi
            question_type = "genel"
            if any(domain in user_input.lower() for domain in self.knowledge_base["domain_knowledge"].keys()):
                for domain in self.knowledge_base["domain_knowledge"].keys():
                    if domain in user_input.lower():
                        question_type = domain
                        break
            
            context = f"""
BAĞLAM BİLGİSİ:
- Soru türü: {question_type}
- İlgili alan bilgisi: {self.knowledge_base["domain_knowledge"].get(question_type, "Genel bilgi")}
"""
        
        return context
    
    def generate_response(self, task: str, user_input: str) -> Dict:
        """Context Engineering ile yanıt üretir"""
        start_time = time.time()
        
        try:
            # Bağlam bilgisi al
            context = self.get_relevant_context(task, user_input)
            
            # Basit prompt + bağlam
            prompt = f"""{context}

GÖREV: {task}
GİRDİ: {user_input}

Yukarıdaki bağlam bilgisini kullanarak yanıt ver:"""
            
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            end_time = time.time()
            
            return {
                "method": "Context Engineering",
                "response": response['message']['content'],
                "context_used": context,
                "response_time": round(end_time - start_time, 2),
                "success": True
            }
            
        except Exception as e:
            return {
                "method": "Context Engineering",
                "response": f"Hata: {str(e)}",
                "response_time": 0,
                "success": False
            }

class ComparisonFramework:
    """İki yöntemi karşılaştırmak için framework"""
    
    def __init__(self, model_name: str = "gemma3n:e4b"):
        self.prompt_eng = PromptEngineering(model_name)
        self.context_eng = ContextEngineering(model_name)
        self.results = []
    
    def run_comparison(self, task: str, test_inputs: List[str]) -> Dict:
        """Karşılaştırma testini çalıştırır"""
        comparison_results = {
            "task": task,
            "timestamp": datetime.now().isoformat(),
            "test_results": []
        }
        
        for i, user_input in enumerate(test_inputs, 1):
            print(f"\n{'='*60}")
            print(f"TEST {i}: {task.upper()}")
            print(f"{'='*60}")
            print(f"GİRDİ: {user_input}")
            print(f"{'-'*60}")
            
            # Prompt Engineering testi
            if task == "sentiment_analysis":
                prompt = self.prompt_eng.create_detailed_prompt(task, user_input)
            else:
                prompt = self.prompt_eng.create_detailed_prompt(task, user_input)
            
            prompt_result = self.prompt_eng.generate_response(prompt)
            
            # Context Engineering testi
            context_result = self.context_eng.generate_response(task, user_input)
            
            # Sonuçları göster
            print(f"\n🎯 PROMPT ENGINEERING:")
            print(f"Yanıt: {prompt_result['response']}")
            print(f"Süre: {prompt_result['response_time']} saniye")
            
            print(f"\n🔗 CONTEXT ENGINEERING:")
            print(f"Yanıt: {context_result['response']}")
            if 'context_used' in context_result:
                print(f"Kullanılan Bağlam: {context_result['context_used'][:100]}...")
            print(f"Süre: {context_result['response_time']} saniye")
            
            test_result = {
                "input": user_input,
                "prompt_engineering": prompt_result,
                "context_engineering": context_result
            }
            
            comparison_results["test_results"].append(test_result)
        
        self.results.append(comparison_results)
        return comparison_results
    
    def analyze_results(self) -> Dict:
        """Sonuçları analiz eder"""
        if not self.results:
            return {"error": "Henüz test sonucu yok"}
        
        total_prompt_time = 0
        total_context_time = 0
        prompt_success = 0
        context_success = 0
        total_tests = 0
        
        for comparison in self.results:
            for test in comparison["test_results"]:
                total_tests += 1
                
                if test["prompt_engineering"]["success"]:
                    prompt_success += 1
                    total_prompt_time += test["prompt_engineering"]["response_time"]
                
                if test["context_engineering"]["success"]:
                    context_success += 1
                    total_context_time += test["context_engineering"]["response_time"]
        
        analysis = {
            "toplam_test": total_tests,
            "prompt_engineering": {
                "başarı_oranı": f"{(prompt_success/total_tests)*100:.1f}%",
                "ortalama_süre": f"{total_prompt_time/max(prompt_success,1):.2f} saniye"
            },
            "context_engineering": {
                "başarı_oranı": f"{(context_success/total_tests)*100:.1f}%",
                "ortalama_süre": f"{total_context_time/max(context_success,1):.2f} saniye"
            }
        }
        
        return analysis

def main():
    """Ana çalıştırma fonksiyonu"""
    print("🚀 PROMPT ENGINEERING vs CONTEXT ENGINEERING KARŞILAŞTIRMASI")
    print("="*70)
    
    # Framework'ü başlat
    comparison = ComparisonFramework("gemma3n:e4b")  # Ollama local model
    
    # Test senaryoları
    test_scenarios = {
        "sentiment_analysis": [
            "Bu ürün gerçekten harika, çok beğendim!",
            "Berbat bir deneyim yaşadım, para israfı",
            "Ortalama bir ürün, fena değil ama süper de değil"
        ],
        "text_summary": [
            """Python programlama dili, 1991 yılında Guido van Rossum tarafından geliştirilmeye başlanmış, 
            yüksek seviyeli, genel amaçlı bir programlama dilidir. Python'un tasarım felsefesi, 
            kod okunabilirliğini vurgular ve özellikle girinti kullanımıyla bunun mümkün olmasını sağlar. 
            Python, çok paradigmalı bir programlama dilidir ve nesne yönelimli, prosedürel ve fonksiyonel 
            programlama stillerini destekler.""",
            
            """Yapay zeka günümüzde birçok alanda kullanılmaktadır. Sağlık sektöründe hastalık teşhisi, 
            finans sektöründe risk analizi, otomotiv endüstrisinde otonom araçlar, e-ticaret platformlarında 
            öneri sistemleri gibi alanlarda aktif olarak kullanılmaktadır. Gelecekte yapay zekanın 
            daha da yaygınlaşacağı ve insan hayatını kolaylaştıracağı öngörülmektedir."""
        ],
        "question_answering": [
            "Python'da liste ve tuple arasındaki fark nedir?",
            "Makine öğrenmesi nedir ve nasıl çalışır?",
            "Sağlıklı beslenme için hangi besinleri tüketmeliyim?"
        ]
    }
    
    # Her senaryo için test çalıştır
    for task, test_inputs in test_scenarios.items():
        print(f"\n{'='*70}")
        print(f"🎯 GÖREV: {task.upper()}")
        print(f"{'='*70}")
        comparison.run_comparison(task, test_inputs)
    
    # Genel analiz
    print(f"\n{'='*70}")
    print("📊 GENEL ANALİZ SONUÇLARI")
    print(f"{'='*70}")
    
    analysis = comparison.analyze_results()
    print(json.dumps(analysis, indent=2, ensure_ascii=False))
    
    # Sonuç yorumu
    print(f"\n{'='*70}")
    print("💡 SONUÇ VE ÖNERİLER")
    print(f"{'='*70}")
    
    print("""
🎯 PROMPT ENGINEERING:
✅ Avantajlar:
   - Detaylı talimatlar ile tutarlı sonuçlar
   - Format kontrolü daha kolay
   - Örneklerle davranış şekillendirme

❌ Dezavantajlar:
   - Uzun promptlar token tüketimi artırır
   - Her görev için yeni prompt tasarımı gerekir
   - Bağlam bilgisi sınırlı

🔗 CONTEXT ENGINEERING:
✅ Avantajlar:
   - Dış kaynaklardan zengin bilgi
   - Dinamik bağlam sağlama
   - Bilgi güncelliği

❌ Dezavantajlar:
   - Ek sistem karmaşıklığı
   - Bağlam seçimi kritik
   - Daha fazla işlem süresi

🏆 GENEL ÖNERİ:
- Basit görevler için: Prompt Engineering
- Bilgi gerektiren görevler için: Context Engineering
- Hibrit yaklaşım genellikle en iyisi
    """)

if __name__ == "__main__":
    main()
