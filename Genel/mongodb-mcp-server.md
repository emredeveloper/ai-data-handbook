**Proje Tanıtımı 🌟**

Bu proje, MongoDB veritabanıyla çalışan uygulamalar geliştirmek ve veritabanı işlemlerini kolaylaştırmak amacıyla oluşturulmuştur. Cursor IDE ile entegre edilen MCP sunucusu, veritabanı işlemlerini daha verimli hale getirir.

---

**MCP'yi Cursor IDE'ye Ekleme Adımları 📋**

Aşağıdaki adımları izleyerek MongoDB MCP sunucusunu Cursor IDE'ye kolayca entegre edebilirsiniz:

1. **MongoDB MCP Sunucusunu Kurun 🖥️**

   Terminalde aşağıdaki komutu çalıştırarak MCP sunucusunu yükleyin:

   ```bash
   npm install -g mongodb-mcp-server
   ```

2. **MCP Sunucusunu Başlatın ▶️**

   MCP sunucusunu başlatmak için şu komutu kullanın:

   ```bash
   npx mongodb-mcp-server --connectionString "mongodb://127.0.0.1:27017/"
   ```

3. **Cursor IDE'de MCP Sunucusunu Tanımlayın ⚙️**

   - Cursor IDE'de `.cursor/mcp.json` dosyasını açın veya oluşturun.
   - Aşağıdaki yapılandırmayı ekleyin:

     ```json
     {
       "mcpServers": {
         "mongodb": {
           "command": "npx",
           "args": [
             "-y",
             "mongodb-mcp-server",
             "--connectionString",
             "mongodb://127.0.0.1:27017/"
           ]
         }
       }
     }
     ```

4. **Cursor IDE'yi Yeniden Başlatın 🔄**

   - Yapılandırmayı kaydettikten sonra Cursor IDE'yi yeniden başlatın.
   - Artık MongoDB işlemlerini IDE üzerinden gerçekleştirebilirsiniz! 🎉

---

**Sonuç 🏁**

Bu entegrasyon, MongoDB veritabanı işlemlerinde hız ve kolaylık sağlar. Cursor IDE ve MCP sunucusu ile geliştirme süreçleriniz daha verimli hale gelir! 
