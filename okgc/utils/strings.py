import spacy


def normalize_lowercase(s: str) -> str:
    """
    Convert a title case string without spaces to lowercase with spaces.
    Also replaces underscores with spaces.
    Examples:
        "Academic Institution" -> "academic institution"
        "AcademicInstitution" -> "academic institution"
        "Academic_Institution" -> "academic institution"
    """
    result = ""
    for i, char in enumerate(s):
        # Replace underscores with spaces
        if char == "_":
            result += " "
            continue
        # Add space before uppercase letters (except the first character)
        # but not if the previous character was an underscore (already added space)
        # or another uppercase letter
        if (
            i > 0
            and char.isupper()
            and s[i - 1] not in [" ", "_", "-", ".", ",", ";", ":"]
            and not s[i - 1].isupper()
        ):
            result += " "
        result += char.lower()
    return result


def simplify_entity_alias(x: str, *, nlp: spacy.Language) -> str:
    # Lemmatization + removal of stopwords + removal of punctuation
    doc = nlp(x)
    z = " ".join(
        token.lemma_ for token in doc if not token.is_stop and not token.is_punct
    )
    if not z:
        # Fallback to just remove punctuation
        z = " ".join(token.text for token in doc if not token.is_punct)
        if not z:
            return x
    # Finally, we ignore any capitalization
    z = z.lower()
    return z
