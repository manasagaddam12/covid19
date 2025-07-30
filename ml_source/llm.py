import google.generativeai as genai
genai.configure(api_key="AIzaSyDPQ-4tzcDURULgIet41lz_iIrJA7-cH1I")
model=genai.GenerativeModel(model_name="gemini-2.0-flash")
chat=model.start_chat(history=[])
while True:
    prmt=input()
    if(prmt=="exit"):
        break
    res=chat.send_message(prmt)
    print(res.text)