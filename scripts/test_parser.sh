#!/usr/bin/env bash
# Compara o comportamento do parser atual com o do legado (tag v0.1-baseline)
# em entradas válidas e malformadas: mesma saída (stdout sem a linha de
# tempo), mesma mensagem de erro e mesmo código de saída.
# Uso: scripts/test_parser.sh ./barenn
set -u
BIN=${1:-./barenn}
DIR=$(mktemp -d)
trap 'rm -rf "$DIR"' EXIT

git show v0.1-baseline:barenn.cpp > "$DIR/legado.cpp"
g++ -O2 -std=c++17 "$DIR/legado.cpp" -o "$DIR/legado"

linha_ok=$(head -1 optdigits.tra)
attrs=$(echo "$linha_ok" | cut -d, -f1-64)
caso() { # nome conteudo...
    local nome=$1; shift
    { cat optdigits.tra | head -20; printf '%b' "$@"; cat optdigits.tra | sed -n '21,40p'; } > "$DIR/$nome.tra"
}
caso faltam_atributos "$(echo "$linha_ok" | cut -d, -f1-63,65)\n"
caso valor_17         "17,${attrs#*,},3\n"
caso valor_negativo   "-1,${attrs#*,},3\n"
caso rotulo_10        "$attrs,10\n"
caso lixo_no_fim      "$attrs,3 x\n"
caso separador_errado "$(echo "$attrs" | sed 's/,/;/5'),3\n"
caso linha_vazia      "\n"
caso espacos_validos  "$(echo "$attrs" | sed 's/,/ , /g') , 3  \n"
caso sinal_mais       "+0,${attrs#*,},+3\n"
tr -d '\r' < optdigits.tra | sed 's/$/\r/' > "$DIR/crlf.tra"
: > "$DIR/vazio.tra"

falhas=0
for f in "$DIR"/*.tra; do
    nome=$(basename "$f" .tra)
    "$DIR/legado" "$f" optdigits.tes > "$DIR/l.out" 2> "$DIR/l.err"; cl=$?
    "$BIN"        "$f" optdigits.tes > "$DIR/n.out" 2> "$DIR/n.err"; cn=$?
    grep -v "^Tempo de treino" "$DIR/l.out" > "$DIR/l.cmp"
    grep -v "^Tempo de treino" "$DIR/n.out" > "$DIR/n.cmp"
    grep -v "^\[fases\]" "$DIR/n.err" | sed "s#$DIR/##" > "$DIR/n.errc"
    sed "s#$DIR/##" "$DIR/l.err" > "$DIR/l.errc"
    if [ $cl -eq $cn ] && cmp -s "$DIR/l.cmp" "$DIR/n.cmp" && cmp -s "$DIR/l.errc" "$DIR/n.errc"; then
        echo "OK     $nome (saida $cl) $(head -c 90 "$DIR/n.errc")"
    else
        echo "FALHA  $nome: legado=$cl novo=$cn"
        diff "$DIR/l.errc" "$DIR/n.errc" | head -4
        falhas=$((falhas + 1))
    fi
done
exit $falhas
