set "DESTIN=%cd%"

set "SOURCE=%cd%\..\ChiLit_Topic_Modeling"

copy "%SOURCE%\data\ChiLit_metadata.csv" "%DESTIN%\data\ChiLit_metadata.csv"
copy "%SOURCE%\data\ChiLit_authors.csv" "%DESTIN%\data\ChiLit_authors.csv"
copy "%SOURCE%\data\ChiLit_Chunks_200.csv" "%DESTIN%\data\ChiLit_Chunks_200.csv"
copy "%SOURCE%\octis_200\Octis_ProdLDA_output.pkl" "%DESTIN%\data\Octis_ProdLDA_output.pkl"
copy "%SOURCE%\octis_200\OCTIS_ProdLDA_Topic_Labels.json" "%DESTIN%\data\Octis_ProdLDA_Topic_Labels.json"