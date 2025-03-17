from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Carregar o modelo treinado e os label encoders
model = joblib.load("model/model.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")

@app.route('/')
def display_gui():
    return render_template('template.html')

@app.route('/verificar', methods=['POST'])
def verificar():
    try:
        # Capturar os dados do formulário
        sexo = request.form['gridRadiosSexo']
        dependentes = request.form['dependentes']
        casado = request.form['gridRadiosCasado']
        trabalho_conta_propria = request.form['gridRadiosTrabalhoProprio']
        rendimento = float(request.form['rendimento'])
        educacao = request.form['educacao']
        valoremprestimo = float(request.form['valoremprestimo'])
        property_area = request.form['property_area']

        # Valores padrão para variáveis que não estão no formulário
        coapplicant_income = 0
        loan_term = 360
        credit_history = 1

        # Criar array de entrada para o modelo
        teste = np.array([[int(sexo), int(casado), int(dependentes), int(educacao), int(trabalho_conta_propria),
                           rendimento, coapplicant_income, valoremprestimo, loan_term, credit_history, int(property_area)]])

        print("Dados enviados para o modelo:", teste)  # Verifica os valores usados

        # Fazer a previsão
        classe = model.predict(teste)[0]

        print("Previsão do modelo:", classe)  # Verifica se a previsão está variando

        return render_template('template.html', classe=str(classe))

    except Exception as e:
        return f"Erro ao processar a requisição: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)
