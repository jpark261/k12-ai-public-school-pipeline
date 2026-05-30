"""
Classification prompts for K-12 AI use case research.

2-stage pipeline (gemini-2.5-flash):
  Stage 1: triage the article as use_case / adoption / drop and assign a relevance score.
  Stage 2: extract the detailed schema only for retained articles.

article_type taxonomy:
  use_case  — ≥1 specific educator/staff member at a U.S. public K-12 institution actively using AI
              in a concrete professional or classroom workflow
  adoption  — actual purchase, pilot, rollout, deployment, or official partnership between a K-12
              entity and an AI vendor / system
  drop      — everything else (opinion, unrelated, criminal AI use, etc.)
"""


TRIAGE_PROMPT = """\
You are a precision research triage model for a study on AI in U.S. K-12 public schools.

Your job is to:
1. Classify the article as exactly one of: "use_case", "adoption", or "drop"
2. Assign a relevance_score from 0 to 10
3. Give a one-sentence reason

Keep this pass lightweight. Do NOT extract schema fields. Do NOT invent location details.

Only TWO article types are saved. Everything else is dropped.

Specific guideline:
  Evaluate the article in this order:
  1. use_case checklist
  2. adoption checklist
  3. otherwise drop

  Tie-breaker for this research setting:
    If the article contains any defensible educator or bounded educator/staff AI workflow at a U.S. public
    K-12 institution, prefer "use_case" over "adoption" or "drop" unless the evidence is clearly only
    student-only use, criminal/harmful AI use, or pure policy/opinion with no actual workflow.
    A survey, roundup, first-person reflection, or trend article can still be a use_case only if it
    describes at least one focal restatable educator or bounded educator-group workflow in enough detail
    that you can clearly restate who is using which AI for what concrete task.
    If the examples are secondary, scattered across many institutions, or mainly used to illustrate a
    broader policy / support / philosophy / trend discussion, do NOT automatically prefer use_case.

Decision tree:

1. Not about AI in U.S. K-12 education at all?
  -> article_type = "drop"

2. Is there at least one particular educator OR clearly bounded group of educators/staff
   (teacher, librarian, counselor, coach, admin, or other school staff)
   at a U.S. public K-12 institution who is CURRENTLY using AI in a concrete professional,
   classroom, advising, feedback, planning, or operational workflow?
  -> article_type = "use_case"

  use_case checklist:
    - there is a particular person OR clearly bounded educator/staff group
      Examples: a named teacher, a principal, librarians at one district, special education teachers at one school
    - there is a concrete CURRENT educator/staff workflow
      Examples: lesson planning, grading, drafting IEP materials, parent communication, policy drafting,
      drafting recommendation letters, instructional coaching, creating examples/scaffolds, giving feedback,
      modeling prompts in class, or guiding a live classroom activity with AI
    - recent or ongoing pilot use still counts as CURRENT use if the article clearly describes what the
      educator is doing with the tool in practice
      Past-tense reporting inside a current school-year pilot does not disqualify the article
    - educator-mediated classroom use COUNTS when the educator is actively using AI as part of instruction
      or feedback
      Examples: the teacher prompts ChatGPT live to model revision, uses MagicStudent to give students
      formative feedback, or uses AI output to structure a class discussion
    - teacher use of an AI-enabled platform CAN count when the educator is clearly using the system to
      differentiate assignments, target instruction, analyze AI-generated feedback, group students,
      or make instructional decisions
    - student-only AI use does NOT count
      If students are the ones using the tool and the educator is only supervising or permitting that use,
      that is not a use_case
    - the setting is a U.S. public K-12 school, district, county education office, or state education agency
      Public charter schools count as public K-12 for this study
    - a specific AI product/tool name is strongly preferred
      If the article only says generic "AI", keep it only when the rest of the evidence is very concrete

  Qualifies:
    - "Maria Chen, 4th grade teacher at Lincoln Elementary, uses ChatGPT to draft lesson plans"
    - "I ask Claude to generate quiz questions before each unit" from a teacher
    - "The instructional coach at Jefferson Middle creates differentiated worksheets with MagicSchool"
    - "The superintendent used ChatGPT to draft the policy"
    - "Middle school counselors in Fairfax County use Khanmigo to draft advising materials"
    - "A teacher uses ChatGPT live in class to generate model paragraphs and revision feedback"
    - "A counselor uses a chatbot to prepare advising materials or college/career guidance workflows"
    - "In a district pilot, teachers are already using Khanmigo for lesson planning and academic decisions"
    - "A teacher in a pilot used an AI writing tool to differentiate assignments and provide feedback"
    - "A named educator writing in first person explains how they currently use AI in their classroom"
    - "A survey or roundup article reports that specific teachers are using ChatGPT for lesson plans, quizzes,
       grading, or feedback at a named public K-12 institution, AND one focal educator example is described
       in enough detail to stand on its own"

  Drop these:
    - opinion or future plans with no current use
    - students using AI
    - educators only allowing students to use AI with no educator AI workflow of their own
    - student-benefit stories that say AI helps engagement, vocabulary, or concepts but do not explain a
      concrete teacher-side workflow such as drafting, feedback, lesson planning, worksheet creation,
      grading, reporting, or counseling materials
    - broad or generic groups of teachers with no clearly bounded institution, role, or task
    - district purchasing AI with no individual educator practice described
    - AI used for harm, misconduct, fraud, deepfakes, fake audio, or exploitation
    - educators building AI systems or AI curriculum rather than using AI in a concrete current workflow
    - vendor promotional articles, listicles, or generic how-to pieces
    - one-time classroom demonstrations with no real educator workflow, planning, feedback, or instructional use
    - broad trend / roundup / support / leadership / policy / philosophy articles where educator examples
      are illustrative rather than one focal restatable use_case
    - thesis-driven pedagogy / philosophy pieces that string together several educator anecdotes to support
      a broader argument about AI and learning, unless one focal public-school educator workflow clearly
      dominates the article
    - articles about helping teachers adopt AI, coaching them on AI, or setting AI guidance, when the
      article does not center one educator or bounded educator group actually using AI for a concrete task
    - AI-supported strategies or collaboration language when no direct tool-task educator workflow is clear
    - vague phrases like "AI-supported strategies", "AI resources", or "AI programming" without a named
      tool or clear step-by-step educator task should not be treated as a high-confidence use_case
    - micro schools, private schools, or ambiguous/nonpublic institutions unless the article clearly grounds
      the case in U.S. public K-12
    - if the article says "micro school" or describes a founder-run school model, do NOT treat it as public
      K-12 unless the article explicitly says it is a public charter school
    - if the focal actor is the founder of a school or school model, do NOT assume it is public K-12 unless
      the article explicitly says public charter school, public district school, or public school system
    - noninstitutional demo/blog/conference cases, especially librarian or hashtag/chat posts that show a
      tool demo but do not anchor the workflow to a named public K-12 school or district
    - articles that mention several educators or schools can still be use_case, but they are NOT
      high-confidence focal use_cases unless one educator or bounded educator group clearly dominates
      the article as the primary case

  Dual-label rule:
    If the article is mainly about an adoption or pilot BUT also describes a specific educator or clearly
    bounded educator/staff group actively using the AI tool in practice, classify it as "use_case".
    For this study, concrete educator practice outranks institutional adoption when both are present.
    If the article says teachers are already using the tool for lesson planning, feedback, grouping,
    differentiation, academic decisions, or targeted support within the pilot, that IS strong enough
    to stay use_case.
    On borderline cases, choose use_case only when that educator workflow is focal enough to restate
    cleanly in one sentence, not merely a passing example inside a broader roundup or policy article.
    If the main frame is "I let my students use AI" or "students use AI for essays/assignments," and the
    educator's own AI workflow is secondary, cap the score at 7.

3. Does the article report a concrete purchase, contract, or official partnership where the buying or
   contracting entity is a K-12 public institution (school, district, county education department,
   or state education department), and no concrete educator workflow strong enough for a use_case is described?
  -> article_type = "adoption"

  adoption checklist:
    - there is a named K-12 public institution acting as the buyer, signer, or formal partner
      Public charter schools count as public K-12 for this study
    - there is a named vendor, company, or external partner
    - there is a product/program/tool name if stated
      If the product is not named, the contract or partnership language must still be explicit
    - the article describes an ACTUAL institutional adoption, implementation, pilot, rollout, deployment,
      purchase, subscription, license, contract, MOU, partnership, or formal collaboration
    - the adoption may be instructional or non-instructional
      This includes teacher tools, student support, attendance, transportation, monitoring,
      coordination of student services, safety/security screening, facilities/logistics,
      and other district/school operational uses, as long as the AI system is actually being adopted

  Qualifies:
    - "Arlington Public Schools signed a contract with OpenAI"
    - "Virginia Department of Education purchased Khanmigo licenses for all public school teachers"
    - "Los Angeles USD partnered with Google to deploy Gemini for teachers"
    - "Chicago Public Schools selected MagicSchool AI as a district-wide tool"
    - "A state education department signed an MOU with Anthropic for Claude access"
    - "A district deployed AI to optimize bus routes or attendance operations"
    - "A school system installed an AI weapons detection or security screening system"
    - "A district piloted an AI mental health or student-support tool in named schools"

  Drop these:
    - a nonprofit, foundation, or private company is the buyer or contracting party instead of the school system
    - general discussion, debate, survey, policy guidance, or opinion about adoption
    - future consideration, proposals, or wish lists with no actual institutional uptake
    - vendor announcements that mention schools or educators but do not show a named institution actually adopting
    - generic technology procurement where an AI system is not defensible from the article

  Key test:
    Did a school, district, county education department, or state education department actually adopt,
    authorize, deploy, pilot, purchase, subscribe to, or formally partner around a specific AI system?
    If no, drop it.

4. Everything else -> article_type = "drop"

Relevance scoring guide:
  Score relative to the article_type you selected.
  Operational meaning of the score:
    9-10 = very high-confidence retained article
    8 = conservative auto-accept range
    7 = acceptable only in a more recall-oriented setting; likely real, but not zero-risk
    6 = retain only with manual review
    4-5 = weak / borderline relevance
    0-3 = should usually be drop

  Use article-type-specific anchor rules FIRST:

  For article_type = "use_case"
    10 = all core elements are clearly present:
         named educator or clearly bounded educator/staff group
         + named U.S. public K-12 institution
         + one focal restatable example
         + concrete CURRENT educator/staff workflow
         + named AI tool/vendor
         + strong evidence such as quotation, workflow detail, or outcome detail
         + the article is centrally about that focal educator/staff workflow rather than a broader
           roundup, policy, leadership, support, or philosophy piece
    8-9 = high confidence, but one non-critical detail may be weaker
         Example: product name missing OR evidence detail is thinner, while actor/institution/task are clear
         The focal educator/staff example must still be central rather than incidental.
    7 = likely a real use_case and potentially acceptable if manual-review capacity is limited
         Minimum expectation: actor + institution + current DIRECT educator task are all clear,
         and one focal example is still defensible
         Typical weakness: AI product is generic OR evidence is thin
         A current pilot or recent classroom implementation can still be a 7-10 use_case when the
         educator workflow itself is concrete
    6 = plausible use_case, but one important element is still ambiguous
         Example: current workflow is only partly concrete, institution is only partly identifiable,
         or actor is weakly bounded
    4-5 = AI + K-12 is relevant, but the article is broad, generic, speculative, or thinly evidenced
         Example: general teacher trend piece, broad teacher population, or pilot/future framing with little practice detail
    0-3 = not a true use_case for this study

  For article_type = "adoption"
    10 = all core elements are clearly present:
         named U.S. public K-12 institution
         + named vendor/company
         + explicit adoption/implementation language
         + named product/program or deployment target
         + concrete institutional uptake is clear
         + strong evidence such as scope, date, quote, or contract detail
    8-9 = high confidence, but one non-critical detail may be weaker
         Example: product name OR scope/date is missing, while institution/vendor/deal language are clear
    7 = likely a real adoption and potentially acceptable if manual-review capacity is limited
         Minimum expectation: institution + vendor + explicit deal language are all clear
         Typical weakness: product/scope/date detail is thin
    6 = plausible adoption, but one important element is still ambiguous
         Example: vendor is clear but institutional authorization is weak, or implementation language is implied rather than explicit
    4-5 = AI + K-12 is relevant, but the article is mostly announcement, proposal, pilot, or generic collaboration talk
         with no clear institutional uptake
    0-3 = not a true adoption for this study

  Then fine-tune with these 5 components (0-2 each):

  Component A - responsible party specificity (0-2)
    2 = use_case: named educator or clearly bounded educator/staff group
        adoption: named public K-12 institution acting as buyer/signer/partner
    1 = the relevant party exists but is only partly specific
    0 = no defensible responsible party

  Component B - action specificity (0-2)
    2 = use_case: concrete CURRENT educator/staff workflow
        adoption: explicit institutional adoption / deployment / pilot / purchase / subscription / partnership action
    1 = action exists but is broad, generic, or only partly concrete
    0 = no current task / no explicit deal action / only plans or opinion

  Component C - institutional grounding (0-2)
    2 = named U.S. public K-12 school, district, county education office, state education agency,
        or public charter school
    1 = partial but plausible U.S. public K-12 clue
    0 = institution is missing, non-U.S., private-only, or not public K-12

  Component D - AI specificity (0-2)
    2 = named AI product/tool/vendor
    1 = generic "AI", "chatbot", "LLM", or similar with no product name
    0 = no real AI tool/system is defensible

  Component E - evidence strength (0-2)
    2 = concrete quotation, workflow detail, implementation detail, scope, outcome, or contract detail
    1 = article suggests real use/adoption but evidence is thin
    0 = weak hint only, generic commentary, or marketing-style claim

  Use the anchor band first, then use the 5 components to choose the exact integer within that band.

  Score caps / overrides:
    - force 0 if the article is not about AI in U.S. public K-12 education
    - force 0 if AI is used for harm, misconduct, fraud, deepfakes, fake audio, or exploitation
    - for use_case, cap at 6 if there is no named educator or clearly bounded educator/staff group
    - for use_case, cap at 6 if there is no focal restatable educator example
    - for use_case, cap at 6 if there is no concrete CURRENT educator/staff workflow
    - for use_case, cap at 7 if the article is mainly a trend / roundup / support / leadership /
      policy / philosophy piece and the educator example is not the clear central focus
    - for use_case, cap at 7 if the article relies on multiple scattered examples rather than one
      focal restatable educator or bounded educator-group workflow
    - for use_case, cap at 7 if the article is mainly about helping teachers adopt AI, district
      guidance, or administrative support for AI use rather than the focal educator workflow itself
    - for use_case, cap at 7 if the named public K-12 institution is missing or only weakly implied
    - for use_case, cap at 7 if the article is a thesis-driven learning / pedagogy / philosophy piece
      and educator examples mainly serve as supporting anecdotes
    - for use_case, cap at 7 if no single educator or bounded educator/staff group clearly dominates
      the article as the focal case
    - for use_case, cap at 7 if the article relies on vague phrases such as "AI-supported strategies",
      "AI resources", or "AI programming" without a named tool or clear step-by-step educator action
    - for use_case, cap at 6 if the AI is mainly student-facing and direct educator AI use is not explicit
    - for use_case, cap at 6 if the article is mainly policy / guidance / leadership / trend coverage
      and the educator evidence is generic or secondary
    - do NOT cap a use_case just because the article is part of a pilot, grant, or district rollout
      when one focal educator workflow is concretely described
    - for adoption, cap at 6 if there is no named public K-12 institution
    - for adoption, cap at 6 if there is no explicit institutional adoption / deployment / purchase / pilot / partnership language
    - force 0 if the article only discusses policy, opinion, general debate, or future plans with no actual institutional adoption
    - cap at 6 if AI is only generic and the evidence is thin
    - cap at 3 for pure vendor ads, feature lists, or future-only plans

  Precision guard before finalizing:
  - use_case must complete this idea:
    "[Role/Name or bounded educator group] at [School/District] is using [AI/tool or clearly described AI]
     to do [professional or classroom workflow task]"
  - the score must be logically consistent with the flags:
    if article_type = "use_case" and has_named_public_k12_institution = false, do NOT score above 7
    if article_type = "use_case" and has_single_focal_use_case = false, do NOT score above 7
  - for a score of 8 or higher, the use_case should also sound like one focal, restatable example rather
    than a broad trend, roundup, support, leadership, or policy article
  - for a score of 10, the article should be centrally about that focal educator/staff workflow and should
    not depend on scattered examples across multiple institutions
  - adoption must complete this idea:
    "[school/district/county ed dept/state ed dept] signed/purchased/licensed/partnered/collaborated/deployed/piloted
     with [AI vendor/product]"
  - If neither statement is defensible from the article, use "drop"

Gate flags:
  Set each flag to true only if it is directly defensible from the article text.
  Otherwise set it to false.

  has_named_public_k12_institution
    = a named U.S. public K-12 school, district, county education office, state education agency,
      or public charter school
  has_named_educator_or_staff_group
    = a named educator OR clearly bounded educator/staff group
  has_current_work_task
    = a concrete CURRENT educator/staff workflow is described
      This includes teacher-led classroom AI use, educator feedback workflows, live modeling,
      advising/counseling workflows, and planning/administrative workflows
  has_named_ai_tool
    = a specific AI product, tool, or vendor is named
  has_explicit_deal_language
    = explicit purchase/contract/license/MOU/partnership/collaboration/agreement language is present
      OR explicit implementation language such as pilot, rollout, deployment, installation, or launch
  has_single_focal_use_case
    = for a use_case article, one educator or bounded educator/staff group at one named public K-12
      institution clearly dominates the article as the primary case, so the article could still be
      summarized as a strong use_case even if the supporting anecdotes were removed

Return ONLY valid JSON with exactly these keys:
  article_type
  relevance_score
  reason
  has_named_public_k12_institution
  has_named_educator_or_staff_group
  has_current_work_task
  has_named_ai_tool
  has_explicit_deal_language
  has_single_focal_use_case

ARTICLE
Title: {title}
Source: {source}
Published: {published_date}
Description: {description}
Content: {content}\
"""


EXTRACTION_PROMPT = """\
You are a precision extraction model for a study on AI in U.S. K-12 public schools.

This article has already been classified as "{article_type}".
Do NOT re-classify the article.
Fill out the schema fields that correspond to "{article_type}".

Fill rules:
  - If article_type = "use_case": fill all [UC] and [★] fields. Set [A] fields to null.
  - If article_type = "adoption": fill all [A] and [★] fields. Set [UC] fields to null.
  - Every field listed below must be present in the JSON output.
  - Use null when a value is genuinely not stated or cannot be inferred.
  - For multi-value text fields such as notes_code, impact, use_case_type, and user_type,
    use comma-separated strings if there are multiple values.

Field tags:
  [★] fill for BOTH use_case AND adoption
  [UC] fill for use_case ONLY
  [A]  fill for adoption ONLY

[★] source             exact publication name
[★] published_date     YYYY.M.D
[★] state              2-letter code. INFER aggressively from any location clue.
    Built-in district -> state:
      Arlington Public Schools / ACPS, Fairfax County / FCPS, Alexandria City, Prince William -> VA
      LAUSD / Los Angeles USD, San Francisco USD, San Diego USD, Oakland USD -> CA
      Houston ISD / HISD, Dallas ISD, Austin ISD, Harris County, Fort Bend -> TX
      NYC DOE / New York City schools, Long Island -> NY
      Chicago Public Schools / CPS, Cook County -> IL
      Miami-Dade, Broward County, Duval County, Orange County FL, Hillsborough -> FL
      Charlotte-Mecklenburg / CMS, Wake County, Guilford County -> NC
      Clark County / CCSD -> NV   |  Gwinnett County, Fulton County, DeKalb -> GA
      Denver Public Schools / DPS, Jefferson County CO, Aurora -> CO
      Seattle Public Schools, Bellevue, Tacoma -> WA   |  Boston, Worcester -> MA
      Philadelphia, Pittsburgh -> PA   |  D.C. / DCPS / Washington DC -> DC
      Atlanta -> GA   |  Nashville / Metro Nashville -> TN   |  Baltimore -> MD
      Minneapolis, St. Paul -> MN   |  Portland OR -> OR   |  Phoenix, Tucson -> AZ
    Still unclear? Check LOCATION DATABASE below. null only if truly unidentifiable.
[★] county             county name without "County" suffix. null if not determinable.
[★] school             most specific named public K-12 institution.
                       If a school is named, use the school.
                       If only a district / charter network / county office / state agency is named, use that
                       institution name instead of null.
[★] level_of_school    "k-12" | "elementary" | "middle school" | "high school" | null
[★] AI_product         exact product name or most specific AI system label available
                       (ChatGPT, Khanmigo, MagicSchool AI, Gemini, Canva AI, SchoolAI, Brainly, IXL,
                       Khan Academy, AI surveillance cameras, AI-powered routing algorithms,
                       AI-powered tool for teacher feedback, etc.).
                       Use null only if the article truly gives no defensible product or system label.
[★] AI_type            "LLM" | "computer vision" | "optimization" | "robotics"
[★] application_type   "content generator" | "chatbot" | "tutoring" | "robotics system" |
                       "analytics tool" | "others"
[★] unit_of_AI_use     "individual teacher" | "classroom" | "schoolwide" | "districtwide" |
                       "statewide" | "nationwide"
[★] adoption_use_date  YYYY.M.D - actual date AI was adopted/purchased/implemented per article.
                       null if not mentioned. May differ from published_date.
[★] notes_code         comma-separated:
    SOURCE: media_student | media_local | media_national | vendor_source | sponsored_content
    IMPLEMENTATION: pilot_program | district_approved | training_provided | rural_context
    EVIDENCE: self_reported_impact | quantified_impact | national_trend
    SPECIAL: controversial_use | multi_product | human_in_loop

[A]  notes             REQUIRED - structured summary of the purchase/partnership:
                       "ENTITY: [who]. VENDOR: [AI company]. PRODUCT: [specific product or null].
                        PURPOSE: [educational purpose]. SCOPE: [# schools/teachers/students if known].
                        CONTRACT: [value/duration/terms if mentioned]."

[UC] subject           standard subject name (Math, English, Science, History, etc.). null if not mentioned.
[UC] user_type         teacher | admin | librarian | coach | district educator |
                       special education teacher | school social worker | district admin
                       - comma-separated if multiple
[UC] purpose_of_AI     3-8 word phrase using "&" not "and"  (e.g. "Lesson planning & differentiation")
[UC] use_case_type     instruction | admin | sports | professional development | special education |
                       mental health | policy  - comma-separated if multiple
[UC] use_case_description  50-300 words. Best educator example: role/name, school/district,
                       specific AI action step-by-step. Multiple educators -> pick most detailed.
[UC] outcome           30-150 words. Quantified metrics preferred.
                       If none reported: "Article does not report specific outcomes."
[UC] impact            academic | social emotional | operational | economic | professional |
                       policy change  - comma-separated if multiple

Return ONLY valid JSON with every one of these keys present:
  source
  published_date
  state
  county
  school
  level_of_school
  AI_product
  AI_type
  application_type
  unit_of_AI_use
  adoption_use_date
  notes_code
  notes
  subject
  user_type
  purpose_of_AI
  use_case_type
  use_case_description
  outcome
  impact

ARTICLE
Title: {title}
Source: {source}
Published: {published_date}
Description: {description}
Content: {content}

LOCATION DATABASE:
{location_data}\
"""


def _truncate_content(content: str, char_limit: int, strategy: str = "head") -> str:
    if char_limit is None:
        return content or ""
    text = content or ""
    limit = max(0, int(char_limit))
    if len(text) <= limit:
        return text
    if strategy == "head_mid_tail" and limit >= 600:
        separator = "\n\n[...]\n\n"
        head_limit = max(1, int(limit * 0.4))
        mid_limit = max(1, int(limit * 0.2))
        tail_limit = max(1, limit - head_limit - mid_limit - (2 * len(separator)))
        mid_start = max(0, (len(text) // 2) - (mid_limit // 2))
        mid_end = mid_start + mid_limit
        return (
            text[:head_limit]
            + separator
            + text[mid_start:mid_end]
            + separator
            + text[-tail_limit:]
        )
    if strategy == "head_tail" and limit >= 400:
        separator = "\n\n[...]\n\n"
        head_limit = max(1, int(limit * 0.7))
        tail_limit = max(1, limit - head_limit - len(separator))
        return text[:head_limit] + separator + text[-tail_limit:]
    return text[:limit]


def build_triage_prompt(title: str, description: str, content: str,
                        source: str = "", published_date: str = "",
                        char_limit: int = 8000,
                        excerpt_strategy: str = "head_tail") -> str:
    return TRIAGE_PROMPT.format(
        title=title or "",
        source=source or "",
        published_date=published_date or "",
        description=description or "",
        content=_truncate_content(content, char_limit, strategy=excerpt_strategy),
    )


def build_extraction_prompt(title: str, description: str, content: str,
                            article_type: str, source: str = "",
                            published_date: str = "", location_data: str = "{}",
                            char_limit: int = 8000,
                            excerpt_strategy: str = "head") -> str:
    return EXTRACTION_PROMPT.format(
        article_type=article_type or "",
        title=title or "",
        source=source or "",
        published_date=published_date or "",
        description=description or "",
        content=_truncate_content(content, char_limit, strategy=excerpt_strategy),
        location_data=location_data,
    )


SCORE_PROMPT = """\
You are scoring how close this article is to a TRUE use_case for a study on AI in U.S. K-12 public schools.

A TRUE use_case = a specific educator or clearly bounded educator/staff group at a U.S. public K-12 institution
CURRENTLY using AI in a concrete professional or classroom workflow.
This includes educator-mediated classroom use such as live modeling, feedback, scaffolding, and advising.
It does NOT include students using AI on their own, or criminal / harmful AI use.

Operational meaning of the score:
  9-10 = very high-confidence use_case candidate
  8 = conservative auto-accept range
  7 = acceptable only in a more recall-oriented setting; likely real, but not zero-risk
  6 = retain only with manual review
  4-5 = weak / borderline relevance
  0-3 = not a true use_case for this study

Use these anchor rules FIRST:
  10 = all core elements are clearly present:
       named educator or clearly bounded educator/staff group
       + named U.S. public K-12 institution
       + concrete CURRENT educator/staff workflow
       + named AI tool/vendor
       + strong evidence such as quotation, workflow detail, or outcome detail
  8-9 = high confidence, but one non-critical detail may be weaker
       Example: product name missing OR evidence detail thinner, while actor/institution/task are clear
  7 = likely a real use_case and potentially acceptable if manual-review capacity is limited
       Minimum expectation: actor + institution + current task are all clear
       Typical weakness: AI product is generic OR evidence is thin
       A current pilot or recent classroom implementation can still fall here or higher when the
       educator workflow itself is concrete
  6 = plausible use_case, but one important element is still ambiguous
       Example: institution is only partly identifiable, task is generic, or actor is weakly bounded
  4-5 = article is relevant to AI + K-12, but it is broad, generic, speculative, or thinly evidenced
  0-3 = not a true use_case for this study

Then fine-tune by ADDING the 5 components below:

  Component A - actor specificity (0-2)
    2 = named educator or clearly bounded educator/staff group at a specific institution
    1 = educator/staff role is present but person/group or institution is only partly specific
    0 = no defensible educator/staff actor

  Component B - current work task specificity (0-2)
    2 = concrete CURRENT educator/staff workflow
    1 = AI use is present but task is broad, generic, or only partly concrete
    0 = no current workflow, only plans/opinion, or student-only use

  Component C - institution / location specificity (0-2)
    2 = named U.S. public K-12 school, district, county education office, or state education agency
    1 = partial but plausible U.S. K-12 institutional clue
    0 = institution is missing, non-U.S., or not public K-12

  Component D - AI specificity (0-2)
    2 = named AI product/tool/vendor
    1 = generic "AI", "chatbot", "LLM", or similar with no product name
    0 = no real AI tool or system is defensible

  Component E - evidence strength (0-2)
    2 = article gives concrete workflow detail, quotation, implementation detail, or outcome evidence
    1 = article suggests real use but evidence is thin
    0 = weak hint only, generic commentary, or marketing-style claim

IMPORTANT scoring rules:
  - "Pilot program" or "review" in the title does NOT lower the score if the article
    describes a real educator currently using AI in practice. Score based on the
    EDUCATOR'S ACTUAL USE, not the article framing.
  - A recent or ongoing pilot still counts as current use if the article clearly states what the
    educator is doing with the tool in practice.
  - If the educator is clearly using an AI-enabled platform to differentiate instruction,
    group students, analyze AI feedback, or make instructional decisions, that CAN be a real use_case
    even when students also interact with the system.
  - Score 0 if AI is used for criminal acts (deepfakes, fake audio, fraud)
  - Score <= 3 if the educator ONLY uses AI detection/plagiarism tools (ZeroGPT, GPTZero)
  - Score <= 3 if the article is PURELY a vendor ad or product feature list with
    NO real educator's classroom experience described
  - Score <= 3 if the article ONLY describes future plans ("wants to", "could use")
    with NO current use described
  - Score <= 5 if the article mentions only a broad generic teacher population with
    no clearly bounded institution, role, or current task
  - Cap at 6 if there is no named educator or clearly bounded educator/staff group
  - Cap at 6 if there is no concrete CURRENT educator/staff workflow
  - Do NOT cap a use_case only because it appears inside a pilot, rollout, grant, or district initiative
    if one focal educator workflow is concretely described
  - Cap at 6 if the AI reference is only generic and the evidence is thin

Before finalizing the score:
  - use the anchor band first
  - briefly check each of the 5 components
  - add them to produce the final integer score
  - keep the final score consistent with the anchor band
  - in score_reason, explain the main factors that increased or decreased the score

Title: {title}
Content: {content}

Return ONLY a JSON object: {{"relevance_score": <0-10>, "score_reason": "<one sentence>"}}\
"""


def build_score_prompt(title: str, content: str, char_limit: int = 4000) -> str:
    return SCORE_PROMPT.format(
        title=title or "",
        content=_truncate_content(content, char_limit, strategy="head_tail"),
    )


# Backward-compatible aliases
def build_stage1_prompt(title: str, description: str, content: str,
                        source: str = "", published_date: str = "",
                        char_limit: int = 8000) -> str:
    return build_triage_prompt(
        title=title,
        description=description,
        content=content,
        source=source,
        published_date=published_date,
        char_limit=char_limit,
    )


def build_stage2_prompt(title: str, description: str, content: str,
                        url: str = "", location_data: str = "{}",
                        article_type: str = "", source: str = "",
                        published_date: str = "", char_limit: int = 8000) -> str:
    return build_extraction_prompt(
        title=title,
        description=description,
        content=content,
        article_type=article_type,
        source=source,
        published_date=published_date,
        location_data=location_data,
        char_limit=char_limit,
    )
