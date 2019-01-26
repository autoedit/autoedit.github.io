function scrollIt(destination, duration = 200) {
	// source: https://pawelgrzybek.com/page-scroll-in-vanilla-javascript/

	const start = window.pageYOffset;
	const startTime = 'now' in window.performance ? performance.now() : new Date().getTime();

	const documentHeight = Math.max(document.body.scrollHeight, document.body.offsetHeight, document.documentElement.clientHeight, document.documentElement.scrollHeight, document.documentElement.offsetHeight);
	const windowHeight = window.innerHeight || document.documentElement.clientHeight || document.getElementsByTagName('body')[0].clientHeight;
	const destinationOffset = typeof destination === 'number' ? destination : destination.offsetTop;
	const destinationOffsetToScroll = Math.round(documentHeight - destinationOffset < windowHeight ? documentHeight - windowHeight : destinationOffset);

	if ('requestAnimationFrame' in window === false) {
		window.scroll(0, destinationOffsetToScroll);
		return;
	}

	function scroll() {
		const now = 'now' in window.performance ? performance.now() : new Date().getTime();
		const time = Math.min(1, ((now - startTime) / duration));
		window.scroll(0, Math.ceil((time * (destinationOffsetToScroll - start)) + start));

		if (window.pageYOffset === destinationOffsetToScroll) {
			return;
		}

		requestAnimationFrame(scroll);
	}

	scroll();
}

function preprocess(data, width, height) {
	const dataFromImage = ndarray(new Float32Array(data), [width, height, 4]);
	const dataProcessed = ndarray(new Float32Array(width * height * 3), [1, 3, width, height]);

	ndarray.ops.divseq(dataFromImage, 255.0);

	ndarray.ops.assign(dataProcessed.pick(0, 0, null, null), dataFromImage.pick(null, null, 0));
	ndarray.ops.assign(dataProcessed.pick(0, 1, null, null), dataFromImage.pick(null, null, 1));
	ndarray.ops.assign(dataProcessed.pick(0, 2, null, null), dataFromImage.pick(null, null, 2));

	return dataProcessed.data;
}

function postprocess(data, width, height) {
	const input = ndarray(new Float32Array(data), [1, 3, width, height]);
	const processed = ndarray(new Float32Array(width * height * 4).fill(255), [width, height, 4]);
	ndarray.ops.assign(processed.pick(null, null, 0), input.pick(0, 0, null, null));
	ndarray.ops.assign(processed.pick(null, null, 1), input.pick(0, 1, null, null));
	ndarray.ops.assign(processed.pick(null, null, 2), input.pick(0, 2, null, null));
	ndarray.ops.mulseq(processed, 255.0);
	return new Uint8ClampedArray(processed.data);
}

document.getElementById('arrow').onclick = function(e) {
	scrollIt(
		document.getElementById('section'),
		700,
		'easeOutQuad'
	);
};


const canvas = document.getElementById("canvas");
const ctx = canvas.getContext('2d');
var img = new Image();

function setCanvasHeight() {
	canvas.style.width = "calc(100% - 10px)";
	if (canvas.offsetHeight > 0.5 * window.innerHeight) {
		var calc = 0.5 * window.innerHeight * canvas.width / canvas.height;
		if (calc < window.innerWidth) {
			canvas.style.width = calc;
		}
	}
}

window.onresize = setCanvasHeight;

function act(event, isFirst) {
	var file = event.target.files[0];
	var reader = new FileReader();
	img = new Image();
	reader.onload = function(event) {
		img.src = event.target.result;
		img.onload = function() {
			if (isFirst) {
				document.getElementById("main").style.display = "none";
			} else {
				document.getElementById("card").style.display = "none";
			}
			document.getElementById("supercontainer").style.display = "block";
			document.getElementById("loading").style.display = "block";
			setTimeout(async function() {
				var max = Math.max(img.height, img.width);
				if (max < 512) {
					var scale = 1;
				} else {
					var scale = 960 / max;
				}
				const width = Math.ceil(img.width * scale);
				const height = Math.ceil(img.height * scale);

				canvas.width = width;
				canvas.height = height;
				ctx.drawImage(img, 0, 0, width, height);
				const pixels = preprocess(ctx.getImageData(0, 0, width, height).data, width, height);
				canvas.width = 96;
				canvas.height = 96;
				ctx.drawImage(img, 0, 0, 96, 96);
				const pixels96 = preprocess(ctx.getImageData(0, 0, 96, 96).data, 96, 96);
				canvas.width = width;
				canvas.height = height;

				const inputTensor = new onnx.Tensor(pixels, 'float32', [1, 3, width, height]);
				const inputTensor96 = new onnx.Tensor(pixels96, 'float32', [1, 3, 96, 96]);

				var session = new onnx.InferenceSession({
					backendHint: 'webgl'
				});
				await session.loadModel(blob);
				const output = await session.run([inputTensor, inputTensor96]);
				var outputData = output.values().next().value.data;

				var image = ctx.createImageData(width, height);
				image.data.set(postprocess(outputData, width, height));
				console.log(width, height);
				ctx.putImageData(image, 0, 0);

				document.getElementById("loading").style.display = "none";
				document.getElementById("card").style.display = "block";
				setCanvasHeight();
			}, 100);
		}
	};
	reader.readAsDataURL(file);
}

function blockInput(event) {
	event.preventDefault();
}

function saveCanvas() {
	var image = canvas.toDataURL("image/png").replace("image/png", "image/octet-stream");
	window.location.href = image;
}

var blob = null;

;
(async () => {
	const saveButton = document.getElementById("save-button");
	const fileInput = document.getElementById("file-input");
	const firstFileInput = document.getElementById("file-input-first");
	const firstFileInputLabel = document.getElementById("file-input-first-label");

	firstFileInput.addEventListener("click", blockInput);
	saveButton.addEventListener("click", saveCanvas);

	var xhr = new XMLHttpRequest();
	xhr.open("GET", "http://localhost:8000/generator.onnx");
	xhr.responseType = "blob";
	xhr.onload = function() {
		blob = xhr.response;

		firstFileInput.removeEventListener("click", blockInput);

		firstFileInputLabel.innerHTML = "Select image";
		firstFileInput.classList.remove("loading-button");
		firstFileInput.onchange = function(event) {
			act(event, true);
		};
		fileInput.onchange = function(event) {
			act(event, false);
		};

	}
	xhr.send();

})();
