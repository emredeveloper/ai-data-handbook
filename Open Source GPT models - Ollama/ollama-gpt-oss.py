from ollama import chat
import time
from datetime import datetime
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    colors_available = True
except ImportError:
    colors_available = False
    print("Colorama kütüphanesi bulunamadı. Renklendirme olmadan devam ediliyor...")
    print("Kurulum için: pip install colorama")

def print_separator(char="=", length=60):
    """Güzel bir ayırıcı çizgi yazdırır"""
    if colors_available:
        print(Fore.CYAN + char * length)
    else:
        print(char * length)

def print_header():
    """Güzel bir başlık yazdırır"""
    print_separator("=")
    if colors_available:
        print(Fore.YELLOW + Style.BRIGHT + "🤖 OLLAMA GPT-OSS AI SOHBET ROBOTU 🤖".center(60))
        print(Fore.GREEN + f"📅 Tarih: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        print(Fore.BLUE + "🔧 Model: gpt-oss:20b")
    else:
        print("🤖 OLLAMA GPT-OSS AI SOHBET ROBOTU 🤖".center(60))
        print(f"📅 Tarih: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        print("🔧 Model: gpt-oss:20b")
    print_separator("=")

def print_question(question):
    """Soruyu güzel bir formatta yazdırır"""
    print()
    if colors_available:
        print(Fore.MAGENTA + "❓ SORU:")
        print(Fore.WHITE + Style.BRIGHT + f"   {question}")
    else:
        print("❓ SORU:")
        print(f"   {question}")
    print_separator("-")

def print_answer_header():
    """Cevap başlığını yazdırır"""
    if colors_available:
        print(Fore.GREEN + "🎯 CEVAP:")
    else:
        print("🎯 CEVAP:")

def print_thinking():
    """Düşünme animasyonu"""
    if colors_available:
        print(Fore.YELLOW + "🤔 AI düşünüyor", end="")
    else:
        print("🤔 AI düşünüyor", end="")
    
    for i in range(3):
        time.sleep(0.5)
        print(".", end="", flush=True)
    print("\n")

# Ana program
print_header()

# Soru
user_question = 'Why is the sky blue?'
print_question(user_question)

# Düşünme animasyonu
print_thinking()

# Cevap başlığı
print_answer_header()

# AI'dan cevap al
try:
    stream = chat(
        model='gpt-oss:20b',
        messages=[{'role': 'user', 'content': user_question}],
        stream=True,
        think=False
    )

    # Cevabı güzel formatta yazdır
    answer_text = ""
    for chunk in stream:
        content = chunk['message']['content']
        answer_text += content
        if colors_available:
            print(Fore.WHITE + content, end='', flush=True)
        else:
            print(content, end='', flush=True)
    
    print("\n")
    print_separator("-")
    
    # Özet bilgiler
    word_count = len(answer_text.split())
    char_count = len(answer_text)
    
    if colors_available:
        print(Fore.CYAN + f"📊 Cevap İstatistikleri:")
        print(Fore.YELLOW + f"   • Kelime sayısı: {word_count}")
        print(Fore.YELLOW + f"   • Karakter sayısı: {char_count}")
    else:
        print("📊 Cevap İstatistikleri:")
        print(f"   • Kelime sayısı: {word_count}")
        print(f"   • Karakter sayısı: {char_count}")

except Exception as e:
    if colors_available:
        print(Fore.RED + f"❌ Hata oluştu: {str(e)}")
    else:
        print(f"❌ Hata oluştu: {str(e)}")

finally:
    print_separator("=")
    if colors_available:
        print(Fore.GREEN + Style.BRIGHT + "✅ Sohbet tamamlandı!")
    else:
        print("✅ Sohbet tamamlandı!")
    print()