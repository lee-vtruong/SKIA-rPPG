#!/bin/bash
FOLDERS=(
"1q6U7jKRBUtr6N2rVs1vwZ2wuUT2VDoRy"
"1tJdI-E143hW0q0NHxmPPkLcr0UvZgeNz"
"1vjXMUmi9CDWvAnrLNoteFNMoxU48Dhzt"
"1y9wb_8Vn1bt34gSMBuc7ux987AzpNEeg"
"1yK1wZdt33J6OxBSyv8kRtNieaH0SgSpi"
"1yR3hc7nfVqJt8VsXz9AIM1r2R5H-pdGC"
"1qCvEJjNRfMVuz4VoQgbDS-RMvgh5Hn9a"
"1qRx5WQvLOwiHM_4w09aKqSLeWVbG4qHE"
"1qavAF-3T7IQdnytHSxqCesdfGqzgavlJ"
"1qavAF-3T7IQdnytHSxqCesdfGqzgavlJ"
"1ql-e1MDCCr4VFLEPPbMpAOXPbYCIX95f"
"1rCP6v2FotK0-SVJGSNgjeA6ppbBb3z0L"
"1rGs5PpY62WdHet_PH6fBGqTMVJvxGAzt"
"1rgia82iJ_6igB0U3tj522jxDa0pdqZTz"
"1rjLl2Nm5ZsDLAsSCCYPQCf08HV5JJwYX"
"1rrd41TEOi9H1_k0cV8pQFZURUE5yS5pb"
"1s46xn-OMg1pEVXnMvl3yB1bJfwBmMX-g"
"1sJXdmzk9mTG_kpCjo00lPkcDKyo5Aqhs"
"1sWjC0FyshavhK8ztwg4I95bF4QU6nybu"
"1soR7P6uArXdJEeaxER7bBKTPuzlhsq1b"
)

for id in "${FOLDERS[@]}"; do
    gdown --folder "https://drive.google.com/drive/folders/${id}" -O ../data/UBFC-rPPG/
done