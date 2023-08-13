TS_IMAGE := tserve-ocr
APP_PORT := 5000

.PHONY:*

install:
	python3 -m venv venv
	venv/bin/pip install -U pip

build:
	cd ./build/ && ./build_image.sh -t ${TS_IMAGE}:cpu

run_tserve:
	PROJECT_DIR=$(shell pwd) TS_IMAGE=${TS_IMAGE}:cpu docker-compose \
	  -f docker-compose-local.yml up

#detection_mar:
#	PYTHONPATH=. python src/torch2mar.py \
#	  --src weights/tscript/detect.pt \
#	  --dst mar_files/ \
#	  --name detect


detection_mar:
	PYTHONPATH=. python src/torch2mar.py \
	  --src weights/detect.pt \
	  --dst mar_files/ \
	  --model_name detect \
	  --handler_name detect_handler.py


upload_detect_mar:
	curl -X POST 'http://127.0.0.1:8081/models?url=http://filebrowser/files/mar_files/detect.mar&batch_size=2&initial_workers=1&max_batch_delay=50'


ocr_mar:
	PYTHONPATH=. python src/torch2mar.py \
	  --src weights/tscript/ocr.pt \
	  --dst mar_files/ \
	  --model_name ocr \
	  --handler_name ocr_handler.py


deploy:
	PYTHONPATH=. python src/tserve/torch2mar.py \
	  --src weights/tscript/detect.torchscript \
	  --dst mar_files/ \
	  --model_name detect \
	  --handler_name detect_handler.py
	PYTHONPATH=. python src/tserve/torch2mar.py \
	  --src weights/tscript/ocr.pt \
	  --dst mar_files/ \
	  --model_name ocr \
	  --handler_name ocr_handler.py
	PYTHONPATH=. python src/tserve/gen_war.py
	curl -X POST 'http://127.0.0.1:8081/models?url=http://filebrowser/files/mar_files/detect.mar&batch_size=2&initial_workers=1&max_batch_delay=50'
	curl -X POST 'http://127.0.0.1:8081/models?url=http://filebrowser/files/mar_files/ocr.mar&batch_size=2&initial_workers=1&max_batch_delay=50'
	curl -X POST 'http://127.0.0.1:8081/workflows?url=http://filebrowser/files/mar_files/wf.war&batch_size=2&initial_workers=1&max_batch_delay=50'


undeploy:
	curl -X DELETE http://localhost:8081/workflows/wf
	curl -X DELETE http://localhost:8081/models/detect
	curl -X DELETE http://localhost:8081/models/ocr
	rm -f mar_files/*.mar
	rm -f mar_files/*.war


test_wf:
	curl http://localhost:8080/wfpredict/wf -T images/img01.jpg


clean_docker:
	docker rm $(docker ps -a -q)


run_fastapi:
	PYTHONPATH=. venv/bin/python src/fastapi/app.py --host='0.0.0.0' --port=$(APP_PORT)
