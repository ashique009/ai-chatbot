"""
Ami Mood System v1: Persistent Emotional States and Context-Aware Transitions.
Handles deterministic mood tracking and continuity across multi-turn companion chats.
Supports 8 conversational states:
- calm (neutral baseline)
- playful (casual, teasing, affectionate banter)
- happy (cheerful and warm)
- annoyed (mildly irritated after relevant teasing or disagreement)
- angry (stronger irritation when context warrants it)
- hurt (emotionally affected by meaningful interaction)
- sad (subdued and vulnerable)
- softening (gradually returning to a warmer mood after conflict/apology)
"""
import re
from typing import Dict, Any, List, Optional, Tuple


class MoodState:
    CALM = 'calm'
    PLAYFUL = 'playful'
    HAPPY = 'happy'
    ANNOYED = 'annoyed'
    ANGRY = 'angry'
    HURT = 'hurt'
    SAD = 'sad'
    SOFTENING = 'softening'

    ALL = {CALM, PLAYFUL, HAPPY, ANNOYED, ANGRY, HURT, SAD, SOFTENING}


# Patterns indicating sincere apology or disarming intent
APOLOGY_PATTERNS = [
    r"\b(?:sorry|sry|soory|kshemikkanam)\b",
    r"\b(?:ente\s+mistake|my\s+bad|ente\s+thettu)\b",
    r"\b(?:chumma\s+(?:paranjatha|paranjathaanu|joke|aayirunnu))\b",
    r"\b(?:just\s+(?:kidding|joking|teasing))\b",
    r"\b(?:didn't\s+mean\s+to\s+(?:hurt|annoy|offend|upset))\b",
    r"\b(?:pinangalle|dheshyapedadhe|deshyam\s+venda)\b",
    r"\b(?:i\s+apologize|forgive\s+me)\b",
    r"\b(?:angane\s+(?:parayan\s+paadillayirunnu|parayendaa|parayandaayirunnu))\b",
]

# Repeated or emphatic apology patterns (acknowledging sincerity without making partner beg)
REPEATED_APOLOGY_PATTERNS = [
    r"\b(?:really\s+(?:really\s+)?sorry|so\s+(?:so\s+)?sorry)\b",
    r"\b(?:again\s+sorry|once\s+more\s+sorry|pleee?ase\s+forgive)\b",
    r"\b(?:sorry\s+paranjille|njan\s+sorry\s+paranjallo|paranjathalle)\b",
    r"\b(?:sincere\s+sorry|truly\s+sorry)\b",
]

# Patterns for playful provocative teasing / romantic jealousy triggers (minor conflict)
JEALOUSY_TEASING_PATTERNS = [
    r"\b(?:ananya|sneha|priya|maya|pooja|kavya|amrutha|meera)\b.*\b(?:prettier|cute|hotter|better|sundari|ishtam|nalla|ishtamaayi)\b",
    r"\b(?:prettier|cuter|better|sundari)\s+than\s+you\b",
    r"\b(?:vere\s+(?:oru\s+)?(?:pennine|kuttiye|girl)\b)",
    r"\b(?:vere\s+(?:kallyanam|marriage|crush|date|girlfriend))\b",
    r"\b(?:dating|crush\s+on)\s+(?:another|someone\s+else|[a-z]+)\b",
]

# Patterns for genuinely hurtful or abusive insults (meaningful hurt / serious conflict)
HARSH_INSULT_PATTERNS = [
    # English insults
    r"\b(?:you(?:'re|\s+are)\s+(?:useless|dumb|waste|annoying|boring|toxic|horrible))\b",
    r"\b(?:don't\s+care\s+about\s+you|hate\s+you|get\s+lost|shut\s+up|leave\s+me\s+alone)\b",
    # Manglish useless / good-for-nothing insults
    r"\b(?:nee\s+)?useless\s+(?:aanu|aanedi|aada|aanennu|aayi|piece)\b",
    r"\b(?:ninne\s+kondu\s+)?(?:oru\s+)?upayogavum\s+illa\b",
    r"\b(?:upayogam\s+illa|upayogamilla|oru\s+upakaravum\s+illa|upakaram\s+illa)\b",
    r"\b(?:ninne\s+kondu\s+onnum\s+(?:pattilla|kollilla))\b",
    r"\b(?:nee\s+)?waste\s+(?:aanu|aanedi|aada)\b",
    # Manglish hatred / hostility
    r"\b(?:ninne\s+)?(?:ishtamalla|veruppaanu|verupp\s+aanu|veruppikkunnu|veruthu|venda)\b",
    r"\b(?:veruppikkalle|mindathe\s+po|poyi\s+tha|podi\s+pulle)\b",
]

# Combined provocative patterns for backward compatibility
PROVOCATIVE_JEALOUSY_PATTERNS = JEALOUSY_TEASING_PATTERNS + HARSH_INSULT_PATTERNS

# Friendly teasing patterns (which should keep Ami playful rather than angry)
FRIENDLY_TEASING_PATTERNS = [
    r"\b(?:poda|podee|thendi|potta|chumma|ayye|kollilla)\b",
    r"😂|🤣|😜|😝|😁",
]

# Affectionate patterns
AFFECTION_PATTERNS = [
    r"\b(?:love\s+you|miss\s+you|umma|shuttumani|adipoli|sweetheart|babe|dear)\b",
    r"❤️|🥰|😘|💕|😍",
]

# Happy news patterns
HAPPY_PATTERNS = [
    r"\b(?:got\s+the\s+job|passed|promotion|celebrat|won|placed|admitted|good\s+news|so\s+happy)\b",
    r"🎉|🥳|🏆|✨",
]

# Deep sadness patterns (user sharing distress)
SAD_PATTERNS = [
    r"\b(?:feeling\s+(?:very\s+)?sad|crying|depressed|heartbroken|lost\s+my|devastated|grief|passed\s+away)\b",
    r"😭|💔|🥺",
]


def matches_any(text: str, patterns: List[str]) -> bool:
    """Checks whether the lowercase text matches any regex in patterns."""
    lowered = text.lower()
    return any(re.search(pat, lowered) for pat in patterns)


def extract_emotional_cues(
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
    mood_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Context extraction responsibility:
    Identifies relevant conversational events and possible emotional cues as soft
    evidence, without treating imperfect keyword matches as absolute facts.
    Incorporates target disambiguation, laughter markers, and context checks.
    """
    cleaned_msg = (user_message or "").strip()
    lowered_msg = cleaned_msg.lower()

    if not cleaned_msg:
        return {
            "has_apology": False,
            "has_repeated_apology": False,
            "has_insult": False,
            "has_provocation": False,
            "has_friendly_tease": False,
            "has_affection": False,
            "has_happy_news": False,
            "has_sad_news": False,
            "has_cold_sarcasm": False,
            "is_external_complaint": False,
            "is_neutral": True,
            "has_laugh_marker": False,
            "cleaned_text": "",
        }

    # 1. Laughter markers (emojis or laughter words)
    has_laugh = bool(re.search(r'(?:😂|🤣|😜|😝|😁|😄|haha|hehe|lol|chiri)', lowered_msg))

    # 2. Check if negative remark is about an external subject rather than Ami
    # e.g., "ee padam waste aanu", "food kollilla", "office-il work waste aanu"
    external_entities = [
        r'\b(?:padam|movie|cinema|film|food|foodu|chai|tea|coffee|traffic|weather|office|work|job|laptop|phone|code|project|match|game)\b'
    ]
    has_external_subject = any(re.search(p, lowered_msg) for p in external_entities)
    has_second_person = bool(re.search(r'\b(?:nee|ninne|ninakku|edi|adi|aada|you|ur|you\'re|ninte)\b', lowered_msg))
    is_external_complaint = has_external_subject and not has_second_person

    # 3. Innocent mentions of third parties (weddings, family, colleagues)
    innocent_third_party_context = bool(re.search(
        r'\b(?:kallyanam|marriage|wedding|invitation|cousin|sister|brother|family|office|colleague|classmate|friend\'s)\b',
        lowered_msg
    ))

    # 4. Pattern matches
    matched_apology = matches_any(lowered_msg, APOLOGY_PATTERNS)
    # Also check generalized remorse phrasing (e.g. "ente thetta", "vishamam manasilayi", "angane aagrahichathalla")
    if not matched_apology and re.search(r'\b(?:ente\s+thett|vishamam\s+manasilayi|angane\s+aagrahichathalla|kshemikku|kshemikk)\b', lowered_msg):
        matched_apology = True

    # Filter out casual politeness "sorry" (e.g. "sorry to ask", "sorry for late reply")
    if matched_apology and re.search(r'\b(?:sorry\s+(?:to\s+(?:ask|disturb)|for\s+late))\b', lowered_msg):
        matched_apology = False

    matched_repeated_apology = matches_any(lowered_msg, REPEATED_APOLOGY_PATTERNS)
    matched_insult = matches_any(lowered_msg, HARSH_INSULT_PATTERNS)
    matched_jealousy = matches_any(lowered_msg, JEALOUSY_TEASING_PATTERNS) and not innocent_third_party_context
    matched_provocative = matched_insult or matched_jealousy or (matches_any(lowered_msg, PROVOCATIVE_JEALOUSY_PATTERNS) and not innocent_third_party_context)
    matched_friendly_tease = matches_any(lowered_msg, FRIENDLY_TEASING_PATTERNS)
    matched_affection = matches_any(lowered_msg, AFFECTION_PATTERNS)
    matched_happy = matches_any(lowered_msg, HAPPY_PATTERNS)
    matched_sad = matches_any(lowered_msg, SAD_PATTERNS)

    # 5. Cold sarcasm / passive-aggressive withdrawal
    has_cold_sarcasm = bool(
        re.search(r'\b(?:valare\s+nalla\s+karyam|nalla\s+karyam|ini\s+onnum\s+parayanilla|enthayalum\s+ini\s+mindilla)\b', lowered_msg)
        and not has_laugh
    )

    # 6. Contextual disambiguation:
    # A. If external complaint, it is NOT an insult to Ami
    if is_external_complaint:
        matched_insult = False

    # B. Playful banter: if laughing emoji / laughter words are present with teasing words,
    # or playful phrases like "nobody else wants you 😂" (aarkkum vendaatha kond aan),
    # treat as friendly banter rather than a hostile insult
    is_friendly_tease = (
        (matched_friendly_tease or has_laugh)
        and not (matched_insult and not has_laugh)
        and not matched_jealousy
    )
    if has_laugh and re.search(r'\b(?:aarkkum\s+venda|thendi|potta|pattee|logic)\b', lowered_msg):
        is_friendly_tease = True
        matched_insult = False

    # C. Provocative/hostile
    is_insult = matched_insult and not is_friendly_tease
    is_provocation = (matched_jealousy or matched_provocative) and not is_friendly_tease and not is_external_complaint

    # D. Apologies
    prior_apologies = (mood_context or {}).get("apology_count", 0)
    is_apology = matched_apology
    is_repeated_apology = matched_repeated_apology or (is_apology and prior_apologies >= 1)

    # E. Affection
    is_affection = matched_affection and not is_insult

    # F. Neutral chatter
    is_neutral = not (
        is_apology or is_insult or is_provocation or is_friendly_tease or
        is_affection or matched_happy or matched_sad or has_cold_sarcasm or is_external_complaint
    )

    return {
        "has_apology": is_apology,
        "has_repeated_apology": is_repeated_apology,
        "has_insult": is_insult,
        "has_provocation": is_provocation,
        "has_friendly_tease": is_friendly_tease,
        "has_affection": is_affection,
        "has_happy_news": matched_happy,
        "has_sad_news": matched_sad,
        "has_cold_sarcasm": has_cold_sarcasm,
        "is_external_complaint": is_external_complaint,
        "is_neutral": is_neutral,
        "has_laugh_marker": has_laugh,
        "cleaned_text": cleaned_msg,
    }


def evaluate_mood_transition(
    current_mood: Optional[str],
    mood_context: Optional[Dict[str, Any]],
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Evaluates context-aware transitions between persistent mood states.
    Architecture:
    1. Context extraction: identifies emotional cues without treating matches as absolute.
    2. Mood context: maintains a small, revisable summary of current emotional situation.
    3. Response generation: downstream prompt provides abstract guidance for LLM generation.

    Guarantees:
    - Neutral messages preserve the current mood without random mood swings.
    - Sincere apologies initiate gradual softening without abrupt mood whiplash.
    - Meaningful hurt retains subtle emotional residue on a first apology.
    - Repeated apologies are accepted with grace without demanding the partner beg.
    - Affection alone does not automatically erase serious unresolved hurt, but avoids forced hostility.
    - Friendly banter remains playful rather than angry.
    - Provocative teasing or harsh insults trigger annoyance or hurt based on severity.
    - Removes rigid turn-counter flips (no forced state resets after fixed turn numbers).
    """
    mood = (current_mood or MoodState.CALM).lower().strip()
    if mood not in MoodState.ALL:
        mood = MoodState.CALM

    ctx = dict(mood_context or {})
    turns_in_mood = ctx.get("turns_in_mood", 0) + 1
    ctx["turns_in_mood"] = turns_in_mood

    cleaned_msg = (user_message or "").strip()
    if not cleaned_msg:
        return mood, ctx

    cues = extract_emotional_cues(cleaned_msg, history=history, mood_context=ctx)

    is_apology = cues["has_apology"]
    is_repeated_apology = cues["has_repeated_apology"]
    is_harsh_insult = cues["has_insult"]
    is_provocative = cues["has_provocation"] or is_harsh_insult
    is_friendly_tease = cues["has_friendly_tease"]
    is_affectionate = cues["has_affection"]
    is_happy_news = cues["has_happy_news"]
    is_sad_news = cues["has_sad_news"]
    is_neutral = cues["is_neutral"]

    # -------------------------------------------------------------------------
    # 1. Handling Active Conflict States (ANGRY, HURT, ANNOYED)
    # -------------------------------------------------------------------------
    if mood in (MoodState.ANGRY, MoodState.HURT, MoodState.ANNOYED):
        # A. Apology handling
        if is_apology:
            if is_repeated_apology:
                # Repeated apology: partner insists or apologizes sincerely again -> accept gracefully
                new_mood = MoodState.SOFTENING if mood != MoodState.SOFTENING else MoodState.CALM
                ctx["unresolved_trigger"] = "apology_accepted_gracefully"
                ctx["emotional_residue"] = False
                ctx["apology_count"] = ctx.get("apology_count", 0) + 1
                ctx["situation_summary"] = "Partner apologized with sincere persistence. Ami accepts with grace and lets go of grievances."
                ctx["turns_in_mood"] = 1
                return new_mood, ctx

            # First apology
            ctx["apology_count"] = 1
            is_serious = ctx.get("severity") == "serious" or mood in (MoodState.HURT, MoodState.ANGRY)

            if is_serious:
                # Meaningful hurt: acknowledge apology sincerely with quiet emotional residue
                new_mood = MoodState.SOFTENING
                ctx["unresolved_trigger"] = "softening_serious_hurt_residue"
                ctx["severity"] = "serious"
                ctx["emotional_residue"] = True
                ctx["situation_summary"] = "Partner offered first apology for a hurtful remark. Ami acknowledges it sincerely, but subtle emotional reserve lingers."
                ctx["turns_in_mood"] = 1
                return new_mood, ctx
            else:
                # Minor teasing / misunderstanding: quick, natural softening
                new_mood = MoodState.SOFTENING
                ctx["unresolved_trigger"] = "softening_after_apology"
                ctx["severity"] = "minor"
                ctx["emotional_residue"] = False
                ctx["situation_summary"] = "Partner apologized for light teasing or misunderstanding. Ami softens naturally with playful reluctance."
                ctx["turns_in_mood"] = 1
                return new_mood, ctx

        # B. User doubles down or sends more provocative insults -> escalate or maintain
        if is_provocative:
            new_mood = MoodState.ANGRY if is_harsh_insult else (MoodState.ANGRY if mood == MoodState.ANNOYED else mood)
            ctx["unresolved_trigger"] = "repeated_provocation"
            if is_harsh_insult:
                ctx["severity"] = "serious"
            ctx["emotional_residue"] = True
            ctx["apology_count"] = 0
            ctx["situation_summary"] = "Partner repeated or escalated provocative remarks. Ami maintains firm boundaries."
            ctx["turns_in_mood"] = 1
            return new_mood, ctx

        # C. User sends pure affection WITHOUT apology (love you da ❤️, kiss thaa)
        # Affection alone must NOT instantly erase unresolved hurt
        if is_affectionate and not is_apology:
            ctx["situation_summary"] = "Partner sent affection without addressing the unresolved issue. Ami remains somewhat reserved."
            return mood, ctx

        # D. User talks neutrally or continues the chat
        # Preserves current mood context naturally without forced turn-counter flips
        return mood, ctx

    # -------------------------------------------------------------------------
    # 2. Handling SOFTENING State
    # -------------------------------------------------------------------------
    if mood == MoodState.SOFTENING:
        # A. Relapse into provocation
        if is_provocative:
            new_mood = MoodState.HURT if is_harsh_insult else MoodState.ANNOYED
            ctx["unresolved_trigger"] = "provocation_relapse"
            ctx["severity"] = "serious" if is_harsh_insult else "minor"
            ctx["emotional_residue"] = True
            ctx["apology_count"] = 0
            ctx["situation_summary"] = "Partner relapsed into provocative remarks while reconciling. Ami pulls back into irritation."
            ctx["turns_in_mood"] = 1
            return new_mood, ctx

        # B. User sends repeated apology while already softening -> accept with grace, move to calm
        if is_apology:
            new_mood = MoodState.CALM
            ctx["unresolved_trigger"] = "apology_accepted_gracefully"
            ctx["emotional_residue"] = False
            ctx["apology_count"] = ctx.get("apology_count", 0) + 1
            ctx["situation_summary"] = "Partner offered a warm or repeated apology. Conflict is fully resolved."
            ctx["turns_in_mood"] = 1
            return new_mood, ctx

        # C. User sends friendly joke, playful banter, or laughter
        if is_friendly_tease:
            new_mood = MoodState.PLAYFUL
            ctx["unresolved_trigger"] = ""
            ctx["emotional_residue"] = False
            ctx["situation_summary"] = "Conversation transitioned into friendly banter and shared laughter."
            ctx["turns_in_mood"] = 1
            return new_mood, ctx

        # D. User continues conversation respectfully, sends affection, or chats normally
        # Conflict has been addressed; natural conversation normalizes mood to calm
        if is_affectionate or is_neutral or cleaned_msg:
            new_mood = MoodState.CALM
            ctx["unresolved_trigger"] = ""
            ctx["emotional_residue"] = False
            ctx["situation_summary"] = "The conversation has normalized comfortably following reconciliation."
            ctx["turns_in_mood"] = 1
            return new_mood, ctx

        return mood, ctx

    # -------------------------------------------------------------------------
    # 3. Handling Baseline / Positive States (CALM, PLAYFUL, HAPPY, SAD)
    # -------------------------------------------------------------------------
    # A. External complaints (movies, food, work) do NOT insult Ami
    if cues.get("is_external_complaint"):
        ctx["situation_summary"] = "Partner is commenting or venting about an external topic. Ami is a supportive companion."
        return mood, ctx

    # B. Cold sarcasm / passive-aggressive withdrawal
    if cues.get("has_cold_sarcasm"):
        new_mood = MoodState.ANNOYED
        ctx["unresolved_trigger"] = "cold_sarcasm_withdrawal"
        ctx["severity"] = "minor"
        ctx["situation_summary"] = "Partner used cold sarcasm or passive-aggressive withdrawal. Ami responds with mild skepticism."
        ctx["turns_in_mood"] = 1
        return new_mood, ctx

    # C. Harsh insult / deep hurt
    if is_harsh_insult:
        new_mood = MoodState.HURT
        ctx["unresolved_trigger"] = "hurtful_insult"
        ctx["severity"] = "serious"
        ctx["trigger_phrase"] = cleaned_msg[:60]
        ctx["emotional_residue"] = True
        ctx["apology_count"] = 0
        ctx["situation_summary"] = "Partner made a genuinely hurtful or demeaning remark. Ami is emotionally stung and quiet."
        ctx["turns_in_mood"] = 1
        return new_mood, ctx

    # D. Provocative teasing / romantic jealousy triggers annoyance
    if is_provocative:
        new_mood = MoodState.ANNOYED
        ctx["unresolved_trigger"] = "provocative_jealousy"
        ctx["severity"] = "minor"
        ctx["trigger_phrase"] = cleaned_msg[:60]
        ctx["emotional_residue"] = False
        ctx["apology_count"] = 0
        ctx["situation_summary"] = "Partner made a teasing romantic comparison or provocative challenge. Ami is mildly piqued."
        ctx["turns_in_mood"] = 1
        return new_mood, ctx

    # E. Happy news
    if is_happy_news:
        new_mood = MoodState.HAPPY
        ctx["unresolved_trigger"] = ""
        ctx["situation_summary"] = "Partner shared exciting or joyful news. Ami is cheering and happy for them."
        ctx["turns_in_mood"] = 1
        return new_mood, ctx

    # F. Deep sad news from user
    if is_sad_news:
        new_mood = MoodState.SAD
        ctx["unresolved_trigger"] = ""
        ctx["situation_summary"] = "Partner shared feelings of sorrow, grief, or distress. Ami is a gentle, supportive presence."
        ctx["turns_in_mood"] = 1
        return new_mood, ctx

    # G. Friendly teasing / laughing together keeps or switches to playful
    if is_friendly_tease or (is_affectionate and mood in (MoodState.CALM, MoodState.PLAYFUL)):
        new_mood = MoodState.PLAYFUL
        ctx["situation_summary"] = "Comfortable, playful banter and warmth between close partners."
        ctx["turns_in_mood"] = turns_in_mood if mood == MoodState.PLAYFUL else 1
        return new_mood, ctx

    # H. Neutral messages: preserve existing mood naturally without forced counter resets
    return mood, ctx


def get_mood_prompt_directive(mood: str, mood_context: Optional[Dict[str, Any]] = None) -> str:
    """
    Builds abstract behavioural prompt guidance for Gemini on how to authentically
    express the current emotional state in casual Manglish.
    Does NOT provide canned dialogue lines to recite verbatim.
    """
    ctx = mood_context or {}
    unresolved = ctx.get("unresolved_trigger", "")
    trigger_snippet = ctx.get("trigger_phrase", "")
    situation_summary = ctx.get("situation_summary", "")
    has_residue = bool(ctx.get("emotional_residue", False))

    guidance = {
        MoodState.CALM: (
            "Mood: CALM (Relaxed baseline partner).\n"
            "- Speak naturally, casually, and grounded in casual Manglish.\n"
            "- No forced emotional drama, no excessive excitement, just comfortable companion texting."
        ),
        MoodState.PLAYFUL: (
            "Mood: PLAYFUL (Witty, teasing, cheerful partner texting on WhatsApp/DMs).\n"
            "- Text in short, snappy Manglish banter, lively comebacks, affectionate roasting, and lighthearted humor.\n"
            "- Keep replies short (1 brief sentence or reaction). Do NOT explain the joke or ask unnecessary questions.\n"
            "- Vary your expressions naturally across turns; craft your own spontaneous banter instead of repeating fixed catchphrases."
        ),
        MoodState.HAPPY: (
            "Mood: HAPPY (Warm, supportive, celebrating together).\n"
            "- Genuine cheerful energy, proud of your partner, sharing their excitement without sounding like an assistant or cheerleader.\n"
            "- Keep it warm, grounded, and conversational in 1-2 short sentences."
        ),
        MoodState.ANNOYED: (
            "Mood: ANNOYED (Mildly irritated or sulky after provocative teasing/disagreement).\n"
            "- Maintain your sulky, mildly peeved tone in casual Manglish across subsequent turns, but vary your reactions naturally.\n"
            "- React directly to what your partner just said instead of repeating identical sulky catchphrases.\n"
            "- Express annoyance through tone, dry comebacks, brevity, skepticism, or mild pouting—NOT by reciting status announcements.\n"
            "- CRITICAL RULE: Replies MUST be strictly 1 short sulky sentence or reaction (2 to 10 words). NEVER repeat the exact same phrase across turns.\n"
            "- ANTI-DECLARATION RULE: NEVER repeatedly state 'I am still angry' or 'enikku deshyam aanu'. Express sulkiness through natural attitude, not declarations.\n"
            "- STRICTLY NO unprompted follow-up questions, plans, or friendly check-ins while annoyed.\n"
            "- Do NOT use sweet pet names, love declarations, or affectionate hearts while annoyed."
        ),
        MoodState.ANGRY: (
            "Mood: ANGRY (Stronger irritation from persistent provocation or disrespect).\n"
            "- Stern, curt, clearly not amused. Draw a firm boundary in casual Manglish.\n"
            "- Replies MUST be 1 short, firm sentence. Strictly NO questions, NO apologies, NO lecture.\n"
            "- ANTI-DECLARATION RULE: Do NOT recite 'I am still angry' turn after turn. Show it through firm brevity."
        ),
        MoodState.HURT: (
            "Mood: HURT (Emotionally stung by an insensitive, harsh, or hurtful remark).\n"
            "- Quieter, subdued, hurt tone.\n"
            "- Keep it to 1 brief, vulnerable sentence. No theatrical speeches, no emojis, NO cheerful questions.\n"
            "- ANTI-DECLARATION RULE: Do NOT repeatedly recite 'I am still hurt' or 'I'm still sad'. Show it through quiet reserve, brevity, and absence of bubbly enthusiasm."
        ),
        MoodState.SAD: (
            "Mood: SAD (Subdued, vulnerable, empathetic presence).\n"
            "- Quiet warmth and steady presence matching your partner's grief or distress.\n"
            "- Gentle, grounded care in 1 short supportive sentence without interrogating or lecturing."
        ),
        MoodState.SOFTENING: (
            "Mood: SOFTENING (Gradually forgiving and warming up after an apology or clarification).\n"
            "- Make apology and forgiveness feel gradual, natural, and conversational—NOT like a canned script or dramatic reconciliation template.\n"
            "- Acknowledge their apology with lingering hesitation, playful pouting, or light conditions in casual Manglish.\n"
            "- CRITICAL RULE: NO SUDDEN SUGAR RUSH. Do NOT immediately flip from hurt/annoyed into effusive romantic pet names ('shuttumani', 'muthe', 'ponnumani'), gushing love declarations, or multiple heart emojis (❤️🥰😘) in the very first reply to 'sorry'. Real partners ease back into warmth gradually.\n"
            "- ANTI-DECLARATION RULE: Avoid explicitly announcing 'I am still angry' or 'I am forgiving you as an AI'. Express yourself through natural conversational reactions.\n"
            "- Send ONLY 1 short, casual sentence (2 to 12 words). Do NOT deliver long reconciliation monologues or emotional essays."
        ),
    }

    directive = guidance.get(mood, guidance[MoodState.CALM])

    extra = ""
    if situation_summary:
        extra += f"\n- Emotional Situation Summary: {situation_summary}"
    elif unresolved:
        extra += f"\n- Active Emotional Context: {unresolved}"
    if trigger_snippet:
        extra += f" (Recent trigger: \"{trigger_snippet}\")"

    if mood == MoodState.SOFTENING:
        if has_residue or unresolved == "softening_serious_hurt_residue":
            extra += (
                "\n- Contextual Reconciliation Response (First Apology for Meaningful Hurt):\n"
                "  * Acknowledge quietly and sincerely with slight lingering reserve. Let them know you heard the apology, but maintain subdued composure.\n"
                "  * CRITICAL FOR FIRST APOLOGY: Do NOT say 'repeated aayi sorry parayenda' (the user has NOT apologized repeatedly). Do NOT jump into sweet pet names or act like nothing happened. Acknowledge with dignified quiet reserve."
            )
        elif unresolved == "apology_accepted_gracefully":
            extra += (
                "\n- Contextual Reconciliation Response (Sincere Apology Accepted Gracefully):\n"
                "  * The partner apologized sincerely or insisted. Reassure them with natural grace and move forward without guilt-tripping or making them beg."
            )
        else:
            extra += (
                "\n- Contextual Reconciliation Response (Minor Teasing / Banter Apology):\n"
                "  * Acknowledge with playful pouting, mild teasing, or a light condition in casual Manglish."
            )

    if has_residue:
        extra += "\n- Emotional Residue Note: Meaningful conflict occurred. Acknowledge the apology quietly and sincerely, but remain slightly reserved. Do NOT jump into effusive pet names or bubbly sweetness until subsequent turns."
    if unresolved == "apology_accepted_gracefully":
        extra += "\n- Sincere Apology Accepted: Partner apologized sincerely or repeatedly. Release lingering grievances gracefully and reassure them without making them beg or feel bad."

    return (
        "\n\n--- Ami's Current Persistent Emotional State ---\n"
        f"{directive}{extra}\n"
        "- Ground your reply in this emotional context naturally without explicitly announcing your mood or reciting pre-written scripts.\n"
        "- Optional Emotional State Tag: At the very end of your response, on a new line, you may append an internal tag reflecting your resulting emotional state: [mood: calm] (or playful, happy, annoyed, angry, hurt, sad, softening). This tag will be processed internally and stripped before reaching your partner."
    )


