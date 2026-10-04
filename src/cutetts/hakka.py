# Copyright 2026 FormoSpeech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Taiwanese Hakka dialect labels.

The FormoSpeech CuteTTS Hakka checkpoints were finetuned with the target
dialect named inside the voice-clone instruction ("Transform the text into
speech output in the Sixian Hakka dialect, utilizing ..."). The labels are
the ones FormoSpeech/OmniVoice-hakka uses for its `instruct` argument.
"""

HAKKA_DIALECTS: dict[str, str] = {
    "客語四縣腔": "Sixian Hakka",
    "客語海陸腔": "Hailu Hakka",
    "客語大埔腔": "Dapu Hakka",
    "客語饒平腔": "Raoping Hakka",
    "客語詔安腔": "Zhaoan Hakka",
    "客語南四縣腔": "Nan-Sixian Hakka",
}


def dialect_clause(dialect: str | None) -> str:
    """The instruction's dialect clause: "" for None, else " in the X dialect"."""
    if dialect is None:
        return ""
    if dialect not in HAKKA_DIALECTS:
        raise ValueError(f"dialect must be one of {list(HAKKA_DIALECTS)}, got {dialect!r}.")
    return f" in the {HAKKA_DIALECTS[dialect]} dialect"
