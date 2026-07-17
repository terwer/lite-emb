#!/bin/bash
# ============================================================
# Rerank 对比测试：cosine vs cross-encoder
# 展示两种模式对同一 query 的排序差异
# ============================================================

URL="http://localhost:8000/v1/rerank"
CT="Content-Type: application/json"

echo "========================================"
echo "  Rerank 对比: cosine vs cross-encoder"
echo "========================================"
echo ""

# --- 测试1: 关键词相近但语义无关 ---
echo "--- 测试1: 关键词重叠但语义不相关 ---"
echo "Query: How to lose weight effectively?"
echo "Docs:  [建议方法] [减肥失败] [比特币价格]"
echo ""

COS=$(curl -s -X POST "$URL" -H "$CT" -d '{
  "model":"e5-small",
  "query":"How to lose weight effectively?",
  "documents":[
    "Exercise 30 minutes daily and eat less sugar",
    "My friend tried everything but still failed to lose weight",
    "The price of Bitcoin dropped 5% today"
  ]
}')
echo "[cosine e5-small]"
echo "$COS" | python3 -c "
import sys,json
r=json.load(sys.stdin)
for x in r['results']:
    print(f'  score={x[\"relevance_score\"]:.4f}  doc={x[\"document\"]}')"

echo ""

CROSS=$(curl -s -X POST "$URL" -H "$CT" -d '{
  "model":"bge-reranker-base",
  "query":"How to lose weight effectively?",
  "documents":[
    "Exercise 30 minutes daily and eat less sugar",
    "My friend tried everything but still failed to lose weight",
    "The price of Bitcoin dropped 5% today"
  ]
}')
echo "[cross-encoder bge-reranker-base]"
echo "$CROSS" | python3 -c "
import sys,json
r=json.load(sys.stdin)
for x in r['results']:
    print(f'  score={x[\"relevance_score\"]:.4f}  doc={x[\"document\"]}')"

echo ""
echo "--- 测试2: 中英文混合，语义精确匹配 ---"
echo "Query: 苹果公司最新产品是什么？"
echo "Docs:  [正确] [水果] [无关]"
echo ""

COS2=$(curl -s -X POST "$URL" -H "$CT" -d '{
  "model":"e5-small",
  "query":"苹果公司最新产品是什么？",
  "documents":[
    "Apple announced the new iPhone 16 with AI features",
    "苹果是一种营养丰富的水果，含有丰富的维生素C",
    "今天天气很好适合出去玩"
  ]
}')
echo "[cosine e5-small]"
echo "$COS2" | python3 -c "
import sys,json
r=json.load(sys.stdin)
for x in r['results']:
    print(f'  score={x[\"relevance_score\"]:.4f}  doc={x[\"document\"]}')"

echo ""

CROSS2=$(curl -s -X POST "$URL" -H "$CT" -d '{
  "model":"bge-reranker-base",
  "query":"苹果公司最新产品是什么？",
  "documents":[
    "Apple announced the new iPhone 16 with AI features",
    "苹果是一种营养丰富的水果，含有丰富的维生素C",
    "今天天气很好适合出去玩"
  ]
}')
echo "[cross-encoder bge-reranker-base]"
echo "$CROSS2" | python3 -c "
import sys,json
r=json.load(sys.stdin)
for x in r['results']:
    print(f'  score={x[\"relevance_score\"]:.4f}  doc={x[\"document\"]}')"

echo ""
echo "========================================"
echo "  解读: cross-encoder 分差更大，更能区分"
echo "  认知: cosine 迅速、免费；cross-encoder 精准"
echo "========================================"
