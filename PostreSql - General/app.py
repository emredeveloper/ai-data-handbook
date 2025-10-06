import psycopg2
import os
from dotenv import load_dotenv

# .env dosyasından ortam değişkenlerini yükle
load_dotenv()

# Bağlantı bilgilerini .env dosyasından al
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")

# Bağlantı dizisi
conn_string = f"dbname={DB_NAME} user={DB_USER} password={DB_PASS} host={DB_HOST}"

try:
    # Veritabanına bağlan
    with psycopg2.connect(conn_string) as conn:
        print(f"'{DB_NAME}' veritabanına başarıyla bağlanıldı!")

        # Cursor oluştur (sorgu çalıştırmak için)
        with conn.cursor() as cur:
            # PostgreSQL versiyonunu al
            cur.execute("SELECT version()")
            db_version = cur.fetchone()
            print(f"PostgreSQL Versiyonu: {db_version[0]}")
            print("-" * 80)
            
            # Veritabanındaki tüm tabloları listele
            cur.execute("""
                SELECT table_name, table_type 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            tables = cur.fetchall()
            
            if tables:
                print(f"\n📊 '{DB_NAME}' veritabanındaki tablolar:")
                print("-" * 80)
                for table_name, table_type in tables:
                    print(f"🔹 {table_name} ({table_type})")
                    
                    # Her tablo için satır sayısını al
                    cur.execute(f"SELECT COUNT(*) FROM {table_name};")
                    count = cur.fetchone()[0]
                    print(f"   └── Satır sayısı: {count}")
                    
                    # Tablo sütun bilgilerini al
                    cur.execute(f"""
                        SELECT column_name, data_type, is_nullable
                        FROM information_schema.columns 
                        WHERE table_name = '{table_name}'
                        ORDER BY ordinal_position;
                    """)
                    columns = cur.fetchall()
                    print(f"   └── Sütunlar: {len(columns)} adet")
                    for col_name, data_type, nullable in columns:
                        null_info = "NULL" if nullable == "YES" else "NOT NULL"
                        print(f"       • {col_name} ({data_type}) - {null_info}")
                    
                    # Eğer tablo boş değilse, ilk 3 satırı göster
                    if count > 0:
                        cur.execute(f"SELECT * FROM {table_name} LIMIT 3;")
                        sample_data = cur.fetchall()
                        print(f"   └── Örnek veriler (ilk 3 satır):")
                        for i, row in enumerate(sample_data, 1):
                            print(f"       {i}. {row}")
                    print()
            else:
                print(f"\n❌ '{DB_NAME}' veritabanında hiç tablo bulunamadı.")

except psycopg2.OperationalError as e:
    print(f"Bağlantı hatası: {e}")

# Bağlantı otomatik olarak kapanacaktır (with bloğu sayesinde)