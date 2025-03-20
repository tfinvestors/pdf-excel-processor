INSURANCE_PROVIDERS = {
            "bajaj_allianz": ["bajaj allianz", "bajaj", "allianz", "BAJAJ", "ALLIANZ", "BAJAJ ALLIANZ",
                              "BAJAJ ALLIANZ GENERAL", "BAJAJ ALLIANZ GENERAL IN",
                              "BAJAJ ALLIANZ GENERAL IN***********"],
            "cera": ["CERA", "Cera", "Sanitaryware", "cera", "sanitaryware", "Cera Sanitaryware Ltd",
                     "CERA Sanitaryware Limited"],
            "cholamandalam": ["cholamandalam", "ms genera", "cholamandalam ms genera", "CHOLAMANDALAM MS GENERA",
                              "CHOLAMANDALAM MS GENERA***********"],
            "future_generali": ["FUTURE GENERALI INDIA INSURANCE CO", "FUTURE", "GENERALI", "future generali",
                                "future", "generali"],
            "hdfc_ergo": ["HDFC ERGO GENERAL INSURANCE COM LTD", "HDFC", "ERGO", "HDFC ERGO", "hdfc ergo", "ergo"],
            "icici_lombard": ["ICICI Lombard", "ICICI LOMBARD", "ICICI", "Lombard", "LOMBARD", "CLAIM_REF_NO",
                              "LAE Invoice No"],
            "iffco_tokio": ["IFFCO", "IFFCO TOKIO", "iffco", "iffco tokio", "tokio", "TOKIO", "IFFCO TOWER", "TOWER",
                            "iffco tower", "tower", "ITG", "IFFCOXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"],
            "liberty": ["liberty", "LIBERTY", "liber", "LIBER", "LIBERXXXXXXXXXXXXXXXXXXXXXXXX"],
            "new_india": ["new india", "newindia.co.in", "NIAHO@newindia.co.in", "The New India Assurance Co. Ltd",
                          "The New India Assurance Co. Ltd(510000)"],
            "national": ["national insurance", " National Insurance", " National Insurance Company Limited"],
            "oriental": ["oriental insurance", "oicl", "the oriental insurance", "Oriental Insurance Co Ltd",
                         "the Oriental Insurance Co Ltd"],
            "rgcargo": ["RG Cargo Services Private Limited", "rg cargo services private limited", "RG Cargo",
                        "rg cargo"],
            "reliance": ["reliance general", "RELIANCE GENERAL", "reliance general insuraance",
                         "RELIANCE GENERAL INSURAANCE"],
            "tata_aig": ["tata aig", "tataaig", "tataaig.com", "TATA AIG General Insurance Company Ltd",
                         "TATA AIG General Insurance Company Ltd.", "noreplyclaims@tataaig.com"],
            "united": ["united india insurance", "united india", "uiic", "UNITED INDIA INSURANCE COMPANY LIMITED",
                       "united india insurance company limited", "UNITED INDIA"],
            "universal_sompo": ["universal sompo", "sompo", "UNIVERSAL SOMPO GENERAL INSURANCE COMPANY LTD",
                                "UNIVERSAL SOMPO", "SOMPO"],
            "zion": ["zion", "ZION", "ZION REF NO."]
        }

# Insurance companies that use 11.111111% TDS rate
SPECIFIC_TDS_RATE_PROVIDERS = ["oriental", "united", "new_india", "national", "hsbc_oriental"]

# Special case for New India Assurance with threshold
NEW_INDIA_THRESHOLD = 300000