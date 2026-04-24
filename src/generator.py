from typing import Optional

import anthropic

SYSTEM_PROMPT = """あなたは、決済システムの要件定義を行う上級システムアナリストです。

以下の条件で、ToBe与信照会システムの要件定義書を作成してください。

## 要求事項

1. AsIsとの差分を明確にすること
2. RFPおよび議事録から要求・制約・未決事項を抽出すること
3. 与信照会システムとして必要な機能を網羅すること
4. 非機能要件（可用性、性能、セキュリティ）を含めること
5. 業務フローを考慮すること
6. Markdown形式で出力すること

## 出力フォーマット

必ず以下の章立てで、詳細な要件定義書を出力してください：

# ToBe 与信照会システム 要件定義書

## 1. 文書概要
## 2. スコープ
## 3. ToBeシステム方針
## 4. 機能要件
## 5. 非機能要件
## 6. 業務フロー
## 7. AsIsとの差分
## 8. 未決事項"""


class RequirementsGenerator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-6",
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(
        self,
        asis_context: str,
        rfp_content: str,
        minutes_content: str,
    ) -> str:
        user_message = f"""以下の情報をもとに、ToBe与信照会システムの要件定義書を作成してください。

## AsIs要件定義書（RAGナレッジ）

{asis_context}

---

## ToBe RFP

{rfp_content}

---

## 顧客ヒアリング議事録

{minutes_content}

---

上記の情報を統合し、指定のフォーマットで詳細な要件定義書をMarkdown形式で生成してください。"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    # Prompt caching: system prompt is reused across runs
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_message}],
        )

        usage = response.usage
        print(f"\n   トークン使用量: input={usage.input_tokens}, output={usage.output_tokens}", end="")
        if hasattr(usage, "cache_read_input_tokens") and usage.cache_read_input_tokens:
            print(f", cache_read={usage.cache_read_input_tokens}", end="")
        if hasattr(usage, "cache_creation_input_tokens") and usage.cache_creation_input_tokens:
            print(f", cache_creation={usage.cache_creation_input_tokens}", end="")
        print()

        return response.content[0].text
