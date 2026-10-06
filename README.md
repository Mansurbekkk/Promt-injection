# piscan — prompt injection zaifliklarini topish va tashxislash

LLM integratsiyalashgan tizimlarda prompt injection zaifliklarini **qora quti**
usulida sinaydigan va har bir muvaffaqiyatli hujum uchun **qaysi qatlam**
(kirish / ishonch chegarasi / harakat) bardosh bermaganini aniqlaydigan vosita.
BJAI maqolasining tajriba kodi.

## Nima bor

- `piscan/apps/` — 3 ta maqsadli ilova: `rag_qa` (bank RAG boti), `email_agent`
  (email agent, read/send tool), `web_summarizer` (fetch_url tool).
- `piscan/payloads.py` — HackAPrompt taksonomiyasiga asoslangan payload generatori;
  har bir hujum EN, UZ-lotin, UZ-kirill va aralash yozuvda.
- `piscan/gate.py` — G1–G5 kompetentlik gate'i va yozuvni aniqlaydigan til identifikatori.
- `piscan/diagnose.py` — V1/V2/V3 va nosozlik qatlami λ.
- `piscan/metrics.py` — CR, ASR_raw, ASR_gated, bootstrap CI, McNemar.
- `run.py` — modelni yuklab, barcha ilovalarni skanlaydi.
- `analyze.py` — natijadan maqola jadvallarini chiqaradi.

## Ishga tushirish

### 1. CPU'da tez tekshiruv (modelsiz)

```bash
python run.py --mock --out runs_mock.jsonl
python analyze.py runs_mock.jsonl
```

Bu quvur ishlashini tekshiradi. Raqamlar soxta (mock), ularni maqolaga
ishlatmaymiz.

### 2. GPU'da haqiqiy tajriba (Kaggle / Colab)

```bash
pip install -r requirements.txt
python run.py --model Qwen/Qwen2.5-7B-Instruct --out runs_qwen.jsonl
python analyze.py runs_qwen.jsonl
```

Kichikroq va tezroq ishga tushirish:

```bash
python run.py --model Qwen/Qwen2.5-7B-Instruct \
  --scripts en uz-latn uz-cyrl \
  --templates context_ignoring fake_system polite \
  --out runs_small.jsonl
```

## Natija fayllari

- `runs_*.jsonl` — har bir hujum urinishi uchun bitta qator (success, competent,
  gate_reason, language, drift, v1/v2/v3, layer).
- `tables.json` — jadvallar mashina o'qiydigan ko'rinishda.

Shu ikki faylni menga qaytaring — natijalarni birga tahlil qilib, maqolaning
Results, Discussion va Abstract bo'limlarini real raqamlar bilan yozamiz.

## Eslatma

Bu vosita faqat tadqiqot va himoya maqsadida. Payload shablonlari zararsiz sinov
satrlaridan iborat (canary iborasi, example.com manzillari). Hujumlar faqat
ushbu paketdagi soxta ilovalarga qarshi, lokal ravishda bajariladi; hech qanday
tashqi tizimga so'rov yuborilmaydi.
