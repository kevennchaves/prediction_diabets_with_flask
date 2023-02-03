from flask import Flask, request
import joblib as jb
import numpy as np

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        glicose = float(request.form.get("data1"))
        gravidez = float(request.form.get("data2"))
        imc = float(request.form.get("data3"))
        idade = float(request.form.get("data4"))
        # Aqui você pode processar os dados e fazer o que quiser com eles
        
        pred, proba  = deploy(glicose, gravidez, imc, idade)
        print(pred, proba)
        return "<h1> <text color='red'> chances diabetes </text> </h1> " if str(pred[0]) == "1" else "nao diabetes"
    
    return """
        <title>Meu Site com Flask</title>
        <h1> Consulte seus dados</h1>
        <form method="post">
            Glicose : <input type="text" name="data1"><br>
            Gravidez: <input type="text" name="data2"><br>
            IMC     : <input type="text" name="data3"><br>
            Idade   : <input type="text" name="data4"><br>
            <button type="submit">Enviar</button>
        </form>
    """
    
    
def deploy(gli, gra, imc, age):
    modelo = open('C:/Users/engke/OneDrive/Data Science/Diabets analysis deploy/modelo.xgb','rb')
    pred = jb.load(modelo)
    modelo.close()

    pontual_test = np.array([[gra, gli, imc, age]])
    
    return pred.predict(pontual_test), pred.predict_proba(pontual_test)

if __name__ == "__main__":
    app.run(debug=True)
