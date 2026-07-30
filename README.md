# AI積読本推薦システム

## 起動

```bash
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## 機能

・蔵書管理

・積読管理

・推薦システム

・読書分析

・Google Books API

## 概要
ローカルのSQLiteデータベースに蔵書を保存し、未読本からTF-IDFとルールベースのスコアでおすすめを表示します。

## 注意
初回起動時に `books.db` が自動生成されます。