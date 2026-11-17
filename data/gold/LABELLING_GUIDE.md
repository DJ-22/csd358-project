# Labelling guide: spelling-variant gold (E1)

*Version 3 (2026-11-22). Annotators label with this version only.*

You will label `candidates.tsv`: 600 groups, each with one **headword** and up to 6 **candidate** spellings taken from the SentiMix training tweets. For every row, decide how the candidate relates to its headword.

Two annotators label the file **independently**. Their agreement (Cohen's κ) is reported, and disagreements are then settled together.

## The three labels

| Label | Meaning | Examples |
|---|---|---|
| `SAME` | The same word in the same grammatical form, spelled differently. | `bahut`/`bohot`, `bht`/`bahut`, `thik`/`theek`, `yaar`/`yar`, `zindagi`/`jindagi`, `achcha`/`achha`, `jhooth`/`jhoot`, `wala`/`vala`, `congress`/`kongress`, `rahul`/`rahool`, `bakwas`/`bakwaas` |
| `INFLECTION` | The same base word in a **different grammatical form**: gender, number, case, tense or person. | `accha`/`acchi`, `bura`/`buri`, `beta`/`beti`, `dekha`/`dekhi`, `ganda`/`gandi`, `dost`/`doston`, `play`/`played` |
| `DIFFERENT` | Different words, however similar they look. | `kal`/`chal`, `kaam`/`naam`, `dil`/`dal`, `sab`/`sub`, `bas`/`bus` when `bus` is the English word, `mat`/`math` |

## How to decide

1. **Read both contexts.** `head_context` and `cand_context` show one real use of each word from the training tweets, with the word in `[brackets]`. First decide **which word each string is, as it is used there**. Tweets swap spellings freely, so the context, not the spelling, tells you the word.
2. **Then ask: would a dictionary list both words under one headword?**
   - One headword, same grammatical form, different spelling → `SAME`.
   - One headword, different grammatical form (gender, number, case, tense, person) → `INFLECTION`.
   - Different headwords → `DIFFERENT`.
3. **Count as `SAME`:**
   - vowel spelled long or short (`aa`/`a`, `ee`/`i`, `oo`/`u`);
   - aspiration written or not (`th`/`t`, `kh`/`k`, `ph`/`f`);
   - doubled or single letters (`sacha`/`sachha`);
   - `w`/`v`, `z`/`j`, `q`/`k`, `c`/`k`, `x`/`ks`;
   - vowels left out (`bht`, `kch`, `pls`);
   - a common abbreviation of the same word (`h`/`hai` only if the contexts clearly show it);
   - spelling variants of the same name;
   - an English loanword written the Hindi way (`kongress`);
   - typos of the same word.
4. **Count as `INFLECTION`:** only when both strings are real forms of one dictionary headword. A typo that happens to change the ending (`politicianss`) is `SAME`.
   - **Pronouns:** case and number forms of one pronoun are `INFLECTION` (`isse`/`isko`, `aapko`/`aapse`, `tum`/`tumhe`, `usko`/`unko`, `is`/`in`, English `he`/`him`). `sab` with a postposition counts as a pronoun (`sabko`/`sabse` → `INFLECTION`). Different pronouns are `DIFFERENT` (`is`/`us`, `tu`/`tum`).
   - **Postpositions:** the agreement forms `ka`/`ki`/`ke` are `INFLECTION`.
   - **Verbs:** any two forms of one verb are `INFLECTION`, including irregular forms: `hai`/`hoon`/`tha`/`thi` (all *hona*), `gaya`/`jaaye` (both *jaana*), `karna`/`kiya`, `raha`/`rahi`.
4b. **Derived words are `DIFFERENT`**, because a dictionary gives them their own entry:
   - noun / adjective: `khush`/`khushi`, `garib`/`garibi`, `dard`/`dardnak`;
   - noun / verb: `bhool`/`bhoolna`, unless the context shows the headword itself used as a verb (rule 1);
   - adverb / related adjective or noun: `aage`/`agla`;
   - plain verb / causative verb: `banna`/`banana`, `dekhna`/`dikhana`;
   - other derivations (prefixes, suffixes such as `-i`, `-ta`, `-ism`).
4c. **Fused phrases.** A string that writes two words as one (`kardo` = *kar do*, `hogaya` = *ho gaya*, `wohi` = *woh hi*, `tabhi` = *tab hi*):
   - fused phrase vs a single word → `DIFFERENT` (`kardo`/`karo`, `wohi`/`woh`, `tabhi`/`tab`);
   - two spellings of the same fused phrase → `SAME` (`hogaya`/`hogya`);
   - the same fused phrase in different grammatical forms → `INFLECTION` (`jarahe`/`jarahi`);
   - a word with an English possessive fused on, vs the word → `DIFFERENT` (`delhis`/`delhi`). Only pronoun case forms (rule 4) are `INFLECTION`.
5. **Homographs** (one spelling, two words, such as `main` = "I" / "in" / English "main", `me`, `mat`):
   - use the sense shown in the context;
   - if the two contexts use different senses → `DIFFERENT`;
   - if the headword in its context is a name or a word from another language (neither Hindi/Urdu nor English), its look-alike candidates are `DIFFERENT`.
6. **Not a word** (a fragment, a stray letter sequence, a username piece) → `DIFFERENT`, unless the context shows it is a shortened form of the headword. A word cut off at the end of a context is `SAME` if the context makes the word clear, otherwise `DIFFERENT`.
7. **Unsure after reading the contexts?** Choose the label you think is most likely, and write a short reason in `note`. Don't leave `label` empty.

## Workflow

1. Copy `candidates.tsv` to your own file:
   - annotator A: `data/gold/variants_gold_A.tsv`
   - annotator B: `data/gold/variants_gold_B.tsv`
2. Fill **only** the `label` column (`SAME`, `INFLECTION` or `DIFFERENT`), and optionally `note`. Don't change, add, remove or reorder any other rows or columns. Keep the file tab-separated and UTF-8. In a spreadsheet, import as text so that no cell is auto-converted.
3. **Work independently.** Don't discuss items with the other annotator until both files are finished. Don't look at any H-Key, Soundex or cluster output while labelling; the file deliberately doesn't say how each candidate was found.
4. When both files are done, run:

   ```
   python experiments/merge_gold.py data/gold/variants_gold_A.tsv data/gold/variants_gold_B.tsv
   ```

   It checks both files, prints Cohen's κ, and writes `data/gold/variants_gold.tsv`. Rows where you agree get that label; rows where you disagree get an empty `label` and show both labels. It also writes `variants_labels_A.tsv` and `variants_labels_B.tsv`: your labels without the context columns.
5. **Adjudicate together:** for every row with an empty `label` in `variants_gold.tsv`, agree on one label and fill it in. Don't change rows that already have a label.

## What gets committed

The context columns are pieces of tweets, and tweet text is not committed (see `data/README.md`).

| File | Tweet text? | In git? |
|---|---|---|
| `candidates.tsv` | yes | **no**; regenerate with `python experiments/e1_variants.py candidates` |
| `variants_gold_A.tsv`, `variants_gold_B.tsv` | yes | **no**; keep them locally |
| `variants_labels_A.tsv`, `variants_labels_B.tsv` | no | yes; `merge_gold.py` accepts them to reproduce κ |
| `variants_gold.tsv` (after adjudication) | no | yes |
| `LABELLING_GUIDE.md` | no | yes |

Keep the three local files out of git with `.git/info/exclude` on your machine:

```
data/gold/candidates.tsv
data/gold/variants_gold_A.tsv
data/gold/variants_gold_B.tsv
```

## How the gold is used

- **Strict score (main result):** a candidate belongs to its headword's group only if labelled `SAME`.
- **Lenient score:** `SAME` or `INFLECTION`.
- Words in different groups count as different words.
- The groups are split into a dev half (to tune the clustering threshold) and a test half (scored once).
