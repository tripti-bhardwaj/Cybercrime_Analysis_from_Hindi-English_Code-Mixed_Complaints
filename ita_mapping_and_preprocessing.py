import re
from collections import defaultdict
from difflib import SequenceMatcher

KEYWORD_SYNONYMS = {
    "hacked": [
        "hackd", "hack", "haking", "hacked into",
        "account hack", "system hack", "phone hack",
        "mobile hack", "device hacked"
    ],
    "unauthorized": [
        "unathorised", "unauthorised",
        "without permission", "without consent",
        "illegal", "illegally", "not authorized"
    ],
    "access": [
        "login", "logged in", "entry", "entered",
        "breach", "intrusion"
    ],
    "transaction": [
        "transction", "txn", "tx",
        "payment", "transfer",
        "upi transaction", "bank transaction"
    ],
    "deducted": [
        "debited", "vanished", "cut",
        "withdrawn", "drained",
        "money gone", "balance gone",
        "paise kat gaye", "amount cut"
    ],
    "bank": [
        "upi", "gpay", "google pay",
        "phonepe", "paytm",
        "netbanking", "internet banking",
        "bank account"
    ],
    "phishing": [
        "fake link", "malicious link",
        "scam link", "fraud link",
        "unknown link", "suspicious link",
        "link pe click", "clicked link"
    ],
    "otp": [
        "one time password", "otp number",
        "verification code", "security code",
        "otp diya", "otp share"
    ],
    "call": [
        "phone call", "missed call",
        "unknown call", "fraud call",
        "scam call", "call aaya"
    ],
    "message": [
        "msg", "sms", "text message",
        "whatsapp message", "telegram message",
        "dm", "direct message"
    ],
    "fake": [
        "fraud", "scam", "bogus",
        "duplicate", "forged",
        "fake profile", "fake account"
    ],
    "impersonation": [
        "posing as", "pretending to be",
        "fake identity", "impersonating",
        "bank executive ban kar"
    ],
    "account": [
        "profile", "id", "user id",
        "login account", "social media account"
    ],
    "password": [
        "passcode", "login password",
        "account password", "pin"
    ],
    "kyc": [
        "know your customer", "kyc update",
        "kyc verification", "kyc process",
        "kyc karwana"
    ],
    "mobile": [
        "phone", "smartphone", "cell phone",
        "android phone", "iphone"
    ],
    "computer": [
        "laptop", "pc", "desktop",
        "system", "machine"
    ],
    "malware": [
        "virus", "trojan", "spyware",
        "ransomware", "worm"
    ],
    "remote access": [
        "anydesk", "teamviewer",
        "screen sharing", "remote control",
        "access diya"
    ],
    "data": [
        "information", "details",
        "personal data", "sensitive data",
        "user data"
    ],
    "leaked": [
        "exposed", "shared",
        "made public", "circulated",
        "viral"
    ]
}

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def levenshtein_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def fuzzy_match(keyword: str, text: str, threshold=0.88) -> bool:
    if len(keyword.split()) > 2:
        return False
    for token in text.split():
        if levenshtein_ratio(keyword, token) >= threshold:
            return True
    return False

def expand_keywords(keywords):
    expanded = set()
    for kw in keywords:
        expanded.add(kw)
        if kw in KEYWORD_SYNONYMS:
            expanded.update(KEYWORD_SYNONYMS[kw])
    return list(expanded)

ITA_RULES = {

    "43": {
        "name": "Penalty for damage to computer, computer system or network",
        "keywords": expand_keywords([
            "money debited", "amount deducted", "balance vanished",
            "balance kam ho gaya", "paise kat gaye", "payment",
            "unknown transaction", "fraud transaction",
            "loss caused due to hacking",
            "computer damage", "system damage", "network damage",
            "system crash", "server crash", "network outage",
            "system slowdown", "system hang ho gaya",
            "screen freeze ho gayi",
            "virus", "malware", "trojan virus",
            "spyware installed", "ransomware attack",
            "virus aa gaya", "malware aa gaya",
            "service disruption", "disruption of computer services",
            "website defacement", "website hacked",
            "data destruction", "data alteration", "data deletion",
            "damage to computer system", "damage to computer network",
            "phone control ho gaya", "data chori ho gaya",
            "mobile slow ho gaya", "system kaam nahi kar raha"
        ])
    },

    "43A": {
        "name": "Failure to protect sensitive personal data",
        "keywords": expand_keywords([
            "data leaked", "personal data misuse", "aadhaar leaked",
            "pan leaked", "bank details leaked", "card details leaked",
            "privacy breach", "company negligence",
            "data breach", "customer data leaked", "user data leaked",
            "database breach", "unauthorized data access",
            "credit card leaked", "debit card leaked",
            "password leaked", "login credentials leaked",
            "biometric data leaked", "violation of privacy",
            "failure to protect data", "lack of data security",
            "inadequate security measures", "poor data protection",
            "negligence in data protection",
            "failure to safeguard data", "data protection failure",
            "non compliance with data protection",
            "sensitive personal data compromised",
            "data leaked due to negligence",
            "unauthorized sharing of personal data",
            "misuse of personal information"
        ])
    },

    "66": {
        "name": "Computer related offences",
        "keywords": expand_keywords([
            "sim", "hacked", "unauthorized", "unauthorized access",
            "illegal access", "unauthorized login",
            "illegal login", "backdoor access",
            "account hacked", "email hacked", "facebook hacked",
            "whatsapp hacked", "gmail hacked", "instagram hacked",
            "social media account hacked",
            "online account compromised",
            "device compromised", "phone compromised",
            "computer misuse", "misuse of computer system",
            "unauthorized use of computer",
            "unauthorized system access",
            "unauthorized modification",
            "phishing link", "clicked link",
            "qr code", "scan qr",
            "anydesk", "remote access",
            "screen sharing", "password changed", "otp intercepted",
            "otp de diya", "otp share ho gaya",
            "account hack ho gaya", "mobile hack ho gaya",
            "phone hack ho gaya", "laptop hack ho gaya",
            "anydesk install karwaya", "remote access diya",
            "link pe click kiya",  "qr scan kiya",
            "computer related offence", "illegal use of computer",
            "unauthorized computer access", "offence under section 66"
        ])
    },

    "66B": {
        "name": "Dishonestly receiving stolen computer resource",
        "keywords": expand_keywords([
            "stolen", "stolen phone",
            "used stolen mobile", "used stolen laptop",
            "stolen device", "illegal device use",
            "stolen computer", "stolen mobile device",
            "stolen electronic device", "stolen digital device",
            "using stolen phone", "using stolen laptop",
            "using stolen computer", "possession of stolen device",
            "dishonestly receiving stolen device", "knowingly using stolen device",
            "illegal possession of device", "unauthorized use of stolen device",
            "stolen phone used for crime", "stolen device misuse",
            "using stolen mobile for fraud", "using stolen device for illegal activity"
        ])
    },

    "66C": {
        "name": "Identity theft",
        "keywords": expand_keywords([
            "aadhaar", "pan card",
            "kyc", "otp shared",
            "cvv", "expiry date",
            "card number", "identity misuse",
            "fake kyc", "password",
            "aadhaar misuse", "pan card misuse",
            "identity theft", "identity fraud",
            "personal details misuse", "credential misuse",
            "otp fraud", "otp misuse",
            "password stolen", "password compromised",
            "login credentials stolen", "credentials misuse",
            "fake identity", "impersonation using documents",
            "forged identity documents", "identity information stolen",
            "bank details misuse", "card details misuse",
            "unauthorized use of identity", "digital identity theft",
            "aadhaar details", "pan details",
            "kyc karwane ko bola", "kyc update bola",
            "otp manga", "otp le liya",
            "password le liya", "details le li",
            "bank details le li", "card details le li",
            "fake kyc", "identity misuse",
            "account mere naam pe", "mere documents misuse"
        ])
    },

    "66D": {
        "name": "Cheating by personation using computer resource",
        "keywords": expand_keywords([
            "msg", "fake call", "fake message",
            "fake customer care", "bank call",
            "pretending to be", "posing as",
            "army man", "jio mart",
            "amazon call", "job offer",
            "naukri call", "insurance call", "impersonation",
            "customer care fraud", "bank impersonation",
            "fake bank executive", "fake company representative",
            "posing as bank official", "posing as government official",
            "online impersonation", "digital impersonation",
            "identity impersonation", "fake identity used",
            "fraudulent phone call", "fraudulent message",
            "scam call", "scam message",
            "job scam", "fake recruitment",
            "employment fraud", "fake offer letter",
            "insurance fraud call", "loan fraud call",
            "fake loan offer", "credit card scam",
            "social media impersonation", "fake profile",
            "posing as friend", "posing as relative",
            "bank se call aaya", "customer care call",
            "army ka bol raha tha", "company ka bol raha tha",
            "amazon se call", "flipkart se call",
            "job dilane ka promise", "job offer aaya",
            "loan dilane ka bola", "insurance ka call",
            "fake call aaya", "scam call",
            "impersonation fraud", "fake profile bana ke"
        ])
    },

    "66E": {
        "name": "Violation of privacy",
        "keywords": expand_keywords([
            "privacy", "private video","private",
            "video leaked", "secret recording", "personal photos",
            "private photo leaked", "private video leaked",
            "personal image leaked", "personal content leaked",
            "unauthorized recording", "hidden camera", "video",
            "secret video recording", "illegal recording",
            "private moments recorded", "intimate video leaked",
            "intimate images leaked", "explicit private content",
            "photo shared without consent", "video shared without consent",
            "unauthorized sharing", "privacy breach",
            "voyeurism", "surveillance without consent",
            "camera misuse", "recorded without permission", "personal video",
            "video viral ho gaya", "photo viral ho gayi",
            "secret recording", "chupke se video",
            "without permission video", "consent ke bina",
            "nude photo", "explicit video",
            "sex video viral", "gandi video",
            "photo share kar diya", "video forward kar diya"
        ])
    },

    "66F": {
        "name": "Cyber terrorism",
        "keywords": expand_keywords([
            "threat to nation", "terror threat",
            "national security", "critical infrastructure attack",
            "threat", "cyber terrorism", "digital terrorism",
            "terrorist cyber attack", "online terror activity",
            "attack on government systems", "attack on defence systems",
            "attack on power grid", "attack on banking infrastructure",
            "national infrastructure attack", "critical systems attack",
            "public utility attack", "communication network attack",
            "threat to sovereignty", "threat to integrity of india",
            "threat to unity of india", "threat to public safety",
            "terrorist organization online", "terror funding online",
            "cyber warfare", "state sponsored cyber attack"
        ])
    },

    "67": {
        "name": "Publishing obscene material",
        "keywords": expand_keywords([
            "obscene video", "pornographic", "picture",
            "explicit content", "nude content",
            "obscene images", "porn video",
            "adult content", "sexually explicit material",
            "pornographic images", "obscene photos",
            "explicit pictures", "adult video",
            "obscene post", "obscene upload",
            "porn shared", "adult content shared",
            "sexual content online", "explicit media",
            "obscene material published", "pornographic content shared",
            "private video", "personal video",
            "video viral ho gaya", "photo viral ho gayi",
            "secret recording", "chupke se video",
            "without permission video", "consent ke bina",
            "nude photo", "explicit video",
            "sex video viral", "gandi video",
            "photo share kar diya", "video forward kar diya"
        ])
    },

    "67A": {
        "name": "Publishing sexually explicit material",
        "keywords": expand_keywords([
            "explicit", "sexual video", "explicit act",
            "sex video", "sexually explicit content",
            "explicit", "digital images", "blackmail", "black mail",
            "digital image", "digital photo",
            "sexually explicit video", "explicit sexual video",
            "sexual act recorded", "explicit sexual act",
            "pornographic video", "hardcore sexual content",
            "explicit adult video", "explicit sexual images",
            "intimate sexual video", "sexual content shared",
            "sexual clip", "adult sexual material",
            "explicit content uploaded", "sexual content published",
            "sex video shared", "explicit media online",
            "private video", "personal video",
            "video viral ho gaya", "photo viral ho gayi",
            "secret recording", "chupke se video",
            "without permission video", "consent ke bina",
            "nude photo", "explicit video",
            "sex video viral", "gandi video",
            "photo share kar diya", "video forward kar diya",
            "video", "chat", "private", "nude"
        ])
    },

    "67B": {
        "name": "Child sexual abuse material",
        "keywords": expand_keywords([
            "child porn", "csam",
            "minor explicit", "underage video",
            "child sexual abuse", "minor sexual content",
            "underage sexual video", "child explicit video",
            "minor porn", "child pornography",
            "sexual content involving minor", "sexual images of child",
            "underage explicit images", "child abuse material",
            "minor sexual exploitation", "child exploitation content",
            "sexual video of minor", "explicit content involving child",
            "child nudity sexual", "underage sexual act",
            "bachche ka video", "minor ka video",
            "underage ladki", "underage ladka",
            "child porn", "bachche ke saath galat",
            "minor ke photo", "bachche ki nude photo",
            "child abuse content", "minor exploitation",
            "schoolgirl", "underaged"
        ])
    }
}

def map_ita_sections(text: str):
    text = normalize_text(text)
    results = []

    for section, rule in ITA_RULES.items():
        hits = 0

        for kw in rule["keywords"]:
            if kw in text or fuzzy_match(kw, text):
                hits += 1

        if hits > 0:
            confidence = min(1.0, hits / max(3, len(rule["keywords"]) * 0.3))
            results.append({
                "it_act_section": section,
                "description": rule["name"],
                "keyword_hits": hits,
                "keyword_confidence": round(confidence, 3)
            })

    return results
