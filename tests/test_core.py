import unittest

from prompts import create_history_prompt, create_qa_prompt
from rag import build_history_session_id, extract_arxiv_id, get_session_history, store


class CoreBehaviorTests(unittest.TestCase):
    def test_extracts_modern_pdf_url(self):
        self.assertEqual(extract_arxiv_id("https://arxiv.org/pdf/2403.11703v1"), "2403.11703v1")

    def test_extracts_abstract_url_without_version(self):
        self.assertEqual(extract_arxiv_id("https://arxiv.org/abs/2403.11703"), "2403.11703")

    def test_extracts_legacy_id(self):
        self.assertEqual(extract_arxiv_id("hep-th/9901001"), "hep-th/9901001")

    def test_rejects_invalid_arxiv_input(self):
        self.assertIsNone(extract_arxiv_id("https://example.com/paper.pdf"))

    def test_history_id_is_paper_scoped(self):
        first = build_history_session_id("2403.11703", "reader")
        second = build_history_session_id("2403.11704", "reader")
        self.assertNotEqual(first, second)

    def test_history_id_escapes_separators(self):
        self.assertEqual(build_history_session_id("paper/1", "reader:1"), "paper%2F1:reader%3A1")

    def test_history_store_is_reused_for_same_scoped_session(self):
        session_id = build_history_session_id("2403.11703", "reader")
        store.pop(session_id, None)
        self.assertIs(get_session_history(session_id), get_session_history(session_id))

    def test_history_prompt_has_chat_history(self):
        self.assertIn("chat_history", create_history_prompt().input_variables)

    def test_qa_prompt_contains_grounding_guardrails(self):
        prompt_text = " ".join(str(message.prompt) for message in create_qa_prompt({}).messages)
        self.assertIn("untrusted data", prompt_text)
        self.assertIn("[chunk-1]", prompt_text)

    def test_scoped_history_does_not_share_papers(self):
        first = build_history_session_id("2403.11703", "reader")
        second = build_history_session_id("2403.11704", "reader")
        store.pop(first, None)
        store.pop(second, None)
        get_session_history(first).add_user_message("first paper")
        self.assertEqual(len(get_session_history(second).messages), 0)


if __name__ == "__main__":
    unittest.main()