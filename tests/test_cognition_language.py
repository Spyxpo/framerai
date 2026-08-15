"""Tests for language and script identification.

Two separate claims are checked, because the module makes two separate claims.
Script detection is expected to work across the world's writing systems - Tamil,
Ge'ez, Hangul, Khmer, Cherokee - and to keep working on text it cannot name a
language for. Language resolution is expected to be right where it has a profile
and *honest* where it does not: an unresolved Cyrillic string comes back as
Cyrillic with low confidence, never as English.

The mind-level tests then check that the language of an experience is remembered
and that the reply is asked for in the same language.
"""

import pytest

from model.cognition import CognitionConfig, Mind, detect_language, detect_script, language_name

# ---------------------------------------------------------------------------
# script coverage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,script",
    [
        ("what is a vocoder", "Latin"),
        ("Що це таке", "Cyrillic"),
        ("Τι είναι αυτό", "Greek"),
        ("ما هو هذا", "Arabic"),
        ("מה זה", "Hebrew"),
        ("यह क्या है", "Devanagari"),
        ("এটা কী", "Bengali"),
        ("இது என்ன", "Tamil"),
        ("ఇది ఏమిటి", "Telugu"),
        ("ಇದು ಏನು", "Kannada"),
        ("ഇത് എന്താണ്", "Malayalam"),
        ("මෙය කුමක්ද", "Sinhala"),
        ("นี่คืออะไร", "Thai"),
        ("ນີ້ແມ່ນຫຍັງ", "Lao"),
        ("នេះជាអ្វី", "Khmer"),
        ("ဒါဘာလဲ", "Myanmar"),
        ("ეს რა არის", "Georgian"),
        ("Սա ինչ է", "Armenian"),
        ("ይህ ምንድን ነው", "Ethiopic"),
        ("ᏙᎯᏧ", "Cherokee"),
        ("이것은 무엇입니까", "Hangul"),
        ("这是什么", "Han"),
        ("これは何ですか", "Japanese"),
    ],
)
def test_script_detection_spans_the_worlds_writing_systems(text, script):
    assert detect_script(text)[0] == script


def test_script_detection_reports_the_dominant_script_of_mixed_text():
    # A Tamil sentence quoting one Latin word is still Tamil.
    script, _, share = detect_script("இந்த model எப்படி வேலை செய்கிறது")
    assert script == "Tamil"
    assert 0.0 < share < 1.0


def test_empty_and_symbol_only_input_is_unidentified():
    assert detect_language("").code == "und"
    assert detect_language("   ").code == "und"
    assert detect_language("123 !!! ###").script == "Unknown"


# ---------------------------------------------------------------------------
# language resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,code",
    [
        ("what is the role of the vocoder in this model", "en"),
        ("¿qué es un vocoder y cómo funciona en el modelo?", "es"),
        ("quel est le rôle du vocodeur dans ce modèle", "fr"),
        ("was ist ein vokoder und wie funktioniert das modell", "de"),
        ("bu bir model ve bu çok daha iyi bir şey için", "tr"),
        ("yang ini tidak dengan model untuk dan di dalam", "id"),
        ("що це таке і як воно працює для його роботи", "uk"),
        ("ما هو هذا وكيف يعمل في هذا النموذج من أجل", "ar"),
        ("این که برای است های خود یک شده مدل", "fa"),
        ("के में की है और से को नहीं यह मॉडल", "hi"),
        ("これは何ですか", "ja"),
        ("이것은 무엇입니까", "ko"),
        ("นี่คืออะไร", "th"),
        ("ეს რა არის", "ka"),
    ],
)
def test_language_resolution_on_languages_with_a_profile(text, code):
    assert detect_language(text).code == code


def test_unresolved_latin_text_admits_it_rather_than_guessing_english():
    # Latin script, no profile matches: the answer is the script, not a language.
    result = detect_language("zzzq wxyv kkjj mmnn ppqq")
    assert result.script == "Latin"
    assert result.code == "und-Latn"
    assert not result.identified
    assert result.confidence < 0.5


def test_confidence_is_lower_for_a_one_word_input_than_a_full_sentence():
    short = detect_language("the")
    full = detect_language("the vocoder is not the part of the model that you asked about")
    assert full.confidence > short.confidence


def test_identified_flag_separates_a_language_from_a_script():
    assert detect_language("this is clearly an english sentence with the words").identified
    assert not detect_language("zzzq wxyv kkjj").identified


def test_language_names_are_available_for_reporting():
    assert language_name("ta") == "Tamil"
    assert language_name("am") == "Amharic"
    assert language_name("xx") == "xx"  # unknown codes pass through unchanged


def test_to_dict_carries_code_name_script_and_confidence():
    data = detect_language("これは何ですか").to_dict()
    assert data["code"] == "ja"
    assert data["name"] == "Japanese"
    assert data["script"] == "Japanese"
    assert 0.0 <= data["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# language in the mind
# ---------------------------------------------------------------------------

def _mind(**overrides) -> Mind:
    base = dict(d_embed=32, episodic_capacity=32, retrieval_k=3, novelty_hidden=32)
    base.update(overrides)
    return Mind(CognitionConfig(**base))


def test_experiences_remember_which_language_they_arrived_in():
    mind = _mind()
    mind.perceive("what is a vocoder and how does it work")
    mind.perceive("これは何ですか")

    languages = {e.language for e in mind.episodic.all()}
    assert languages == {"en", "ja"}
    assert mind.self_model.languages == {"en": 1, "ja": 1}


def test_unidentified_input_is_counted_separately_not_mislabelled():
    mind = _mind()
    mind.perceive("zzzq wxyv kkjj mmnn")
    assert mind.self_model.unidentified_languages == 1
    assert mind.self_model.languages == {}


def test_competence_is_tracked_per_language():
    mind = _mind(sleep_threshold=1.0)
    for i in range(6):
        mind.perceive(f"this is an english sentence about the model number {i}")
    mind.perceive("これは何ですか")

    topics = mind.curiosity.progress.topics()
    assert "lang:en" in topics and "lang:ja" in topics
    assert mind.curiosity.progress.competence("lang:en") > 0.0


def test_context_asks_for_a_reply_in_the_language_it_was_addressed_in():
    mind = _mind()
    trace = mind.perceive("quel est le rôle du vocodeur dans ce modèle")
    assert "French" in mind.context(trace)


def test_context_stays_quiet_when_the_language_is_a_guess():
    mind = _mind()
    trace = mind.perceive("zzzq wxyv kkjj")
    assert "Answer in" not in mind.context(trace)


def test_topic_labels_survive_scripts_that_do_not_space_their_words():
    """Splitting Japanese on whitespace yields one word: the whole sentence."""
    for text in ("これは何ですか", "这是什么模型", "นี่คืออะไร"):
        topic = Mind.infer_topic(text)
        assert 0 < len(topic) <= 3
        assert topic != text


def test_topic_labels_skip_function_words_in_any_language():
    # "qué", "es", "un", "cómo" are grammar; the subject word is what is wanted.
    assert Mind.infer_topic("¿qué es un vocoder y cómo funciona?") not in {"cómo", "qué"}
    assert Mind.infer_topic("was ist ein vokoder und wie funktioniert das") != "funktioniert das"


def test_competence_is_reported_per_language_and_per_sense():
    mind = _mind(sleep_threshold=1.0)
    mind.perceive("this is an english sentence about the model")
    mind.perceive("これは何ですか")

    languages = mind.competence_by("lang")
    assert set(languages) == {"en", "ja"}
    assert all(0.0 <= v <= 1.0 for v in languages.values())
    assert set(mind.competence_by("sense")) == {"text"}


def test_introspect_reports_the_languages_it_has_lived_in():
    mind = _mind()
    mind.perceive("what is a vocoder and how does it work")
    mind.perceive("これは何ですか")
    mind.perceive("이것은 무엇입니까")

    snapshot = mind.introspect()
    assert dict(snapshot["languages"]) == {"en": 1, "ja": 1, "ko": 1}
    assert snapshot["senses"] == {"text": 3}
