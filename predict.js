const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");

const IMAGE_SIZE = 224;

async function predictGender(imagePath) {
  const model = await tf.loadLayersModel("file://./model/model.json");
  const buffer = fs.readFileSync(imagePath);
  const tensor = tf.node
    .decodeImage(buffer)
    .resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE])
    .toFloat()
    .div(255.0)
    .expandDims();

  const prediction = model.predict(tensor);
  const [prob] = await prediction.data();

  let genero = "Não foi possível identificar";
  if (prob > 0.6) genero = "Feminino";
  else if (prob < 0.4) genero = "Masculino";

  return {
    genero,
    confianca: prob.toFixed(2),
  };
}

module.exports = predictGender;
