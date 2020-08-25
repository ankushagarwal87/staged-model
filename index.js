let mobilenet;
const webcam = new Webcam(document.getElementById('wc'));
const enableWebcamButton = document.getElementById('startPredicting');
let isPredicting = false;
const SIGN_CLASSES = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V',
    22: 'W',
    23: 'X',
    24: 'Y',
    25: 'Z'
}
  
// Check if webcam access is supported.
function getUserMediaSupported() {
  return !!(navigator.mediaDevices &&
    navigator.mediaDevices.getUserMedia);
}

// If webcam supported, add event listener to button for when user
// wants to activate it to call enableCam function which we will 
// define in the next step.
if (!getUserMediaSupported()) {
  console.warn('getUserMedia() is not supported by your browser');
}


async function loadMobilenet() {
  console.log('Loading model..');
  const mobilenet = await  tf.loadLayersModel('tfjs_sign_model/model.json');
  console.log('Successfully loaded model');
  //console.log(mobilenet.inputs);
  //console.log(mobilenet.output);
  return tf.model({inputs: mobilenet.inputs, outputs: mobilenet.output});
}

async function predict() {  
  var memory='';
  while (isPredicting) {        
  const img = webcam.capture()  
  let predictions = await mobilenet.predict(img).data();
    let top5 = Array.from(predictions)
    .map(function (p, i) {
        return {
            probability: p,
            className: SIGN_CLASSES[i]
        };
    }).sort(function (a, b) {
        return b.probability - a.probability;
    }).slice(0, 5);
    console.log(top5[0]);
    
    var predictionText = top5[0].className;
    if(top5[0].probability > 0.4 && memory != predictionText){     
     document.getElementById("prediction").innerText = document.getElementById("prediction").innerText + predictionText;
     memory = predictionText;  
    }
    img.dispose();
    //predictions.dispose();
    //top5.dispose();
    await tf.nextFrame();
  
  }
}



function startPredicting(){
	isPredicting = true;
	predict();
}

function stopPredicting(){
	isPredicting = false;
  document.getElementById("prediction").innerText=''
	predict();
}

async function init(){
  mobilenet = await loadMobilenet();
	await webcam.setup();	
}

init();
