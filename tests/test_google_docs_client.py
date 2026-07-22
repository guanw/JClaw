from __future__ import annotations

from typing import Any

from jclaw.tools.google_docs.client import GoogleDocsClient, extract_document_id, normalize_document


class FakeExecute:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response

    def execute(self) -> dict[str, Any]:
        return self.response


class FakeDocumentsResource:
    def __init__(self, service: FakeDocsService) -> None:
        self.service = service

    def get(self, *, documentId: str) -> FakeExecute:
        self.service.calls.append(("get", {"documentId": documentId}))
        return FakeExecute(self.service.document_response)

    def batchUpdate(self, *, documentId: str, body: dict[str, Any]) -> FakeExecute:
        self.service.calls.append(("batchUpdate", {"documentId": documentId, "body": body}))
        return FakeExecute(self.service.batch_update_response)


class FakeDocsService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.document_response = {
            "documentId": "doc-123",
            "title": "Lease Template",
            "revisionId": "rev-1",
            "body": {
                "content": [
                    {
                        "paragraph": {
                            "elements": [
                                {"textRun": {"content": "Tenant: {{tenant_name}}\n"}},
                                {"textRun": {"content": "Rent: {{rent_amount}}\n"}},
                            ]
                        }
                    }
                ]
            },
        }
        self.batch_update_response = {
            "replies": [
                {"replaceAllText": {"occurrencesChanged": 2}},
                {"replaceAllText": {"occurrencesChanged": 1}},
            ],
            "writeControl": {"requiredRevisionId": "rev-2"},
        }

    def documents(self) -> FakeDocumentsResource:
        return FakeDocumentsResource(self)


class FakeFilesResource:
    def __init__(self, service: FakeDriveService) -> None:
        self.service = service

    def copy(self, *, fileId: str, body: dict[str, Any]) -> FakeExecute:
        self.service.calls.append(("copy", {"fileId": fileId, "body": body}))
        return FakeExecute({"id": "copy-123", "name": "Lease Copy"})


class FakeDriveService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def files(self) -> FakeFilesResource:
        return FakeFilesResource(self)


class FakeAuth:
    def __init__(self) -> None:
        self.docs = FakeDocsService()
        self.drive = FakeDriveService()
        self.loaded: list[tuple[str, tuple[str, ...]]] = []
        self.built: list[tuple[str, str, object]] = []

    def load_credentials(self, token_name: str, scopes: tuple[str, ...]) -> object:
        self.loaded.append((token_name, scopes))
        return "creds"

    def build_service(self, service_name: str, version: str, creds: object) -> object:
        self.built.append((service_name, version, creds))
        if service_name == "docs":
            return self.docs
        if service_name == "drive":
            return self.drive
        raise AssertionError(f"unexpected service {service_name}")


def test_extract_document_id_accepts_doc_urls_and_raw_ids() -> None:
    assert extract_document_id("abc123_-") == "abc123_-"
    assert (
        extract_document_id("https://docs.google.com/document/d/1DU3CHO5fYbwbgX_ZFFQpBvoZSsVI8a6Gq9BkIA403MI/edit")
        == "1DU3CHO5fYbwbgX_ZFFQpBvoZSsVI8a6Gq9BkIA403MI"
    )


def test_normalize_document_extracts_text_and_placeholders() -> None:
    normalized = normalize_document(
        {
            "documentId": "doc-1",
            "title": "Template",
            "revisionId": "rev-1",
            "body": {
                "content": [
                    {"paragraph": {"elements": [{"textRun": {"content": "Hello {{name}}\nTenant: [tenant_name]\n"}}]}},
                    {
                        "table": {
                            "tableRows": [
                                {
                                    "tableCells": [
                                        {"content": [{"paragraph": {"elements": [{"textRun": {"content": "{{rent}}"}}]}}]}
                                    ]
                                }
                            ]
                        }
                    },
                ]
            },
        }
    )

    assert normalized["document_id"] == "doc-1"
    assert normalized["title"] == "Template"
    assert normalized["text_preview"] == "Hello {{name}}\nTenant: [tenant_name]\n{{rent}}"
    assert normalized["placeholders"] == ["[tenant_name]", "{{name}}", "{{rent}}"]
    assert normalized["placeholder_count"] == 3
    assert normalized["body_element_count"] == 2


def test_google_docs_client_get_document_uses_docs_api() -> None:
    auth = FakeAuth()
    client = GoogleDocsClient(auth, token_name="docs", scopes=("scope-one",))

    document = client.get_document("doc-123")

    assert document["title"] == "Lease Template"
    assert document["placeholders"] == ["{{rent_amount}}", "{{tenant_name}}"]
    assert auth.loaded == [("docs", ("scope-one",))]
    assert auth.built == [("docs", "v1", "creds")]
    assert auth.docs.calls == [("get", {"documentId": "doc-123"})]


def test_google_docs_client_copy_document_uses_drive_api() -> None:
    auth = FakeAuth()
    client = GoogleDocsClient(auth, token_name="docs", scopes=("scope-one",))

    copied = client.copy_document("doc-123", name="Lease Copy")

    assert copied == {
        "document_id": "copy-123",
        "title": "Lease Copy",
        "url": "https://docs.google.com/document/d/copy-123/edit",
    }
    assert auth.drive.calls == [("copy", {"fileId": "doc-123", "body": {"name": "Lease Copy"}})]


def test_google_docs_client_replace_text_builds_batch_update() -> None:
    auth = FakeAuth()
    client = GoogleDocsClient(auth, token_name="docs", scopes=("scope-one",))

    result = client.replace_text("doc-123", {"{{tenant_name}}": "Alice", "{{rent_amount}}": "$2500"})

    assert result == {
        "document_id": "doc-123",
        "replacement_count": 3,
        "request_count": 2,
        "revision_id": "rev-2",
    }
    assert auth.docs.calls == [
        (
            "batchUpdate",
            {
                "documentId": "doc-123",
                "body": {
                    "requests": [
                        {
                            "replaceAllText": {
                                "containsText": {"text": "{{tenant_name}}", "matchCase": True},
                                "replaceText": "Alice",
                            }
                        },
                        {
                            "replaceAllText": {
                                "containsText": {"text": "{{rent_amount}}", "matchCase": True},
                                "replaceText": "$2500",
                            }
                        },
                    ]
                },
            },
        )
    ]
