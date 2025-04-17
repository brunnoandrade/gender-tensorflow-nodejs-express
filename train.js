const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");

const IMAGE_SIZE = 224;
const BATCH_SIZE = 8;
const EPOCHS = 10;

function loadImagesFromFolder(folderPath, label) {
  const allowedExtensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"];
  const files = fs
    .readdirSync(folderPath)
    .filter((file) =>
      allowedExtensions.includes(path.extname(file).toLowerCase())
    );

  return files.map((file) => ({
    path: path.join(folderPath, file),
    label,
  }));
}

async function loadData() {
  const maleImages = loadImagesFromFolder("data/male", 0);
  const femaleImages = loadImagesFromFolder("data/female", 1);
  const allImages = [...maleImages, ...femaleImages];

  const tensors = [];
  const labels = [];

  for (const img of allImages) {
    try {
      const buffer = fs.readFileSync(img.path);
      const tensor = tf.node
        .decodeImage(buffer)
        .resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE])
        .toFloat()
        .div(255.0)
        .expandDims();
      tensors.push(tensor);
      labels.push(img.label);
    } catch (err) {
      console.warn(`Erro ao carregar ${img.path}: ${err.message}`);
    }
  }

  return {
    xs: tf.concat(tensors),
    ys: tf.tensor(labels, [labels.length, 1]),
  };
}

function createModel() {
  const model = tf.sequential();

  model.add(
    tf.layers.conv2d({
      inputShape: [IMAGE_SIZE, IMAGE_SIZE, 3],
      filters: 32,
      kernelSize: 3,
      activation: "relu",
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  model.add(
    tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: "relu" })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 128, activation: "relu" }));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

  model.compile({
    loss: "binaryCrossentropy",
    optimizer: tf.train.adam(),
    metrics: ["accuracy"],
  });

  return model;
}

(async () => {
  const { xs, ys } = await loadData();
  const model = createModel();

  await model.fit(xs, ys, {
    epochs: EPOCHS,
    batchSize: BATCH_SIZE,
    validationSplit: 0.2,
    callbacks: tf.callbacks.earlyStopping({
      patience: 3,
      restoreBestWeight: true,
    }),
  });

  await model.save("file://./model");
  console.log("✅ Modelo salvo em ./model");
})();
