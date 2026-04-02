async function treinarEPrever() {

    const status = document.getElementById("status");
    const resultadoTexto = document.getElementById("resultado");

    const valor = document.getElementById("temperatura").value;
    const temperatura = Number(valor);

    // =========================
    // VALIDAÇÃO
    // =========================
    if (!valor || isNaN(temperatura)) {
        resultadoTexto.innerText = "⚠️ Digite uma temperatura válida!";
        return;
    }

    // 🚫 BLOQUEIO DE LIMITE
    if (temperatura > 55) {
        resultadoTexto.innerText = "🚫 Temperatura máxima permitida é 55°C!";
        return;
    }

    if (temperatura < 0) {
        resultadoTexto.innerText = "🚫 Temperatura não pode ser negativa!";
        return;
    }

    if (typeof tf === "undefined") {
        status.innerText = "❌ TensorFlow não carregou!";
        return;
    }

    status.innerText = "Treinando IA...";

    // =========================
    // NORMALIZAÇÃO
    // =========================
    const tempMin = 0;
    const tempMax = 40;

    const consumoMin = 1;
    const consumoMax = 5;

    function normalizarTemp(t) {
        return (t - tempMin) / (tempMax - tempMin);
    }

    function desnormalizarConsumo(c) {
        return c * (consumoMax - consumoMin) + consumoMin;
    }

    // =========================
    // DADOS DE TREINO
    // =========================
    const xs = tf.tensor2d([
        normalizarTemp(0),
        normalizarTemp(10),
        normalizarTemp(20),
        normalizarTemp(30),
        normalizarTemp(40)
    ], [5, 1]);

    const ys = tf.tensor2d([
        (1 - consumoMin) / (consumoMax - consumoMin),
        (1.5 - consumoMin) / (consumoMax - consumoMin),
        (2.5 - consumoMin) / (consumoMax - consumoMin),
        (3.5 - consumoMin) / (consumoMax - consumoMin),
        (4.5 - consumoMin) / (consumoMax - consumoMin)
    ], [5, 1]);

    // =========================
    // MODELO
    // =========================
    const modelo = tf.sequential();

    modelo.add(tf.layers.dense({
        units: 8,
        inputShape: [1],
        activation: 'relu'
    }));

    modelo.add(tf.layers.dense({
        units: 1
    }));

    modelo.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'meanSquaredError'
    });

    // =========================
    // TREINAMENTO
    // =========================
    await modelo.fit(xs, ys, {
        epochs: 300,
        shuffle: true
    });

    status.innerText = "IA treinada!";

    // =========================
    // PREVISÃO
    // =========================
    const tempNormalizada = normalizarTemp(temperatura);

    const previsao = modelo.predict(
        tf.tensor2d([tempNormalizada], [1, 1])
    );

    const valorNormalizado = (await previsao.data())[0];

    const consumoFinal = desnormalizarConsumo(valorNormalizado);

    // =========================
    // RESULTADO
    // =========================
    resultadoTexto.innerText =
        `💧 Consumo estimado: ${consumoFinal.toFixed(2)} litros`;
}