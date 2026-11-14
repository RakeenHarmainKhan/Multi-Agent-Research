import pytest

from app.guardrails import redact_pii


@pytest.mark.parametrize(
    "text, expected, kinds",
    [
        ("mail ana.k@corp.co.uk now", "mail [EMAIL] now", ["email"]),
        ("call 415-555-0132", "call [PHONE]", ["phone"]),
        ("call (415) 555-0132", "call [PHONE]", ["phone"]),
        ("call +1 415 555 0132", "call [PHONE]", ["phone"]),
        ("ssn 123-45-6789", "ssn [SSN]", ["ssn"]),
        ("card 4111 1111 1111 1111", "card [CARD]", ["card"]),
        ("card 4111-1111-1111-1111", "card [CARD]", ["card"]),
    ],
)
def test_masks_personal_data(text, expected, kinds):
    assert redact_pii(text) == (expected, kinds)


@pytest.mark.parametrize(
    "text",
    [
        "Paper published 2026-11-14 at 12:44",  # date: the old regex called this a phone number
        "Model trained on 1,200,000,000 tokens",
        "ISBN 978-3-16-148410-0",
        "order 20261114123456",  # long id that fails the card checksum
        "card 4111 1111 1111 1112",  # right shape, wrong checksum
        "ssn 000-12-3456",  # impossible area number
        "version 2.1.0 released",
        "What is LoRA?",
    ],
)
def test_leaves_everything_else_alone(text):
    assert redact_pii(text) == (text, [])


def test_several_kinds_are_listed_once_each_in_order():
    text, kinds = redact_pii("a@b.com, c@d.com and 415-555-0132")
    assert text == "[EMAIL], [EMAIL] and [PHONE]"
    assert kinds == ["email", "phone"]


def test_an_ssn_is_not_mislabelled_as_a_phone_number():
    """Regression for the Rag project's regexes, where the phone pattern swallowed SSNs and cards."""
    assert redact_pii("123-45-6789")[1] == ["ssn"]
