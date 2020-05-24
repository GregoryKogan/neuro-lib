let TrainingData = [
  {
    Inputs: [0, 1],
    Targets: [1]
  },
  {
    Inputs: [1, 0],
    Targets: [1]
  },
  {
    Inputs: [0, 0],
    Targets: [0]
  },
  {
    Inputs: [1, 1],
    Targets: [0]
  }
];

let Specifications;

let Brain;

function setup() {
  createCanvas(windowWidth, windowHeight);
  Specifications = new Specification;
  Specifications.SetOptions("New", undefined, 2, 1, 2, [5, 4]);
  Brain = new NeuralNetwork(Specifications);
  Brain.SetLearningRate(0.1);
}

function draw(){
    for (let i = 0; i < 100; ++i){
      let data = random(TrainingData);
      Brain.Train(data.Inputs, data.Targets);
    }
    background(18);
    textSize(60);
    textAlign(CENTER);
    fill(255);
    text(Math.round(Brain.Predict([1, 0]) * 100000000) / 100000000, windowWidth / 2 + windowWidth / 9, windowHeight / 5);
    text(Math.round(Brain.Predict([0, 1]) * 100000000) / 100000000, windowWidth / 2 + windowWidth / 9, windowHeight / 5 * 2);
    text(Math.round(Brain.Predict([1, 1]) * 100000000) / 100000000, windowWidth / 2 + windowWidth / 9, windowHeight / 5 * 3);
    text(Math.round(Brain.Predict([0, 0]) * 100000000) / 100000000, windowWidth / 2 + windowWidth / 9, windowHeight / 5 * 4);
    textSize(75);
    text("XOR", windowWidth / 5 + windowWidth / 8, windowHeight / 2);
    textSize(15);
    text("GregoNeuron V1.1", windowWidth / 5 + windowWidth / 8, windowHeight / 2 + 25);
}
