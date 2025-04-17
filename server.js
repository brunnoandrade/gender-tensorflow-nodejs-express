const express = require("express");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const predictGender = require("./predict");

const app = express();
const upload = multer({ dest: "uploads/" });

app.post("/upload", upload.single("image"), async (req, res) => {
  try {
    const imagePath = req.file.path;
    const genero = await predictGender(imagePath);
    fs.unlinkSync(imagePath);
    res.json({ genero });
  } catch (err) {
    res
      .status(500)
      .json({ erro: "Erro ao classificar imagem", detalhes: err.message });
  }
});

app.listen(3000, () => {
  console.log("🚀 Servidor rodando em http://localhost:3000");
});
