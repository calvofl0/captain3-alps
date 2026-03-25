# The most recent PyTorch container from NVIDIA that is compatible with the driver version 550.54.15 available on Piz Daint as of 2026-03-26 is `nvidia/pytorch:24.06-py3`
# It features Python 3.10, while CAPTAIN v.3.0 requires Python>=3.11
# Some hack is hence required, part of it is made adding constraints on the versions of some dependencies
FROM nvcr.io/nvidia/pytorch:24.06-py3
LABEL org.opencontainers.image.authors="Flavio.Calvo@unil.ch"

USER root

ARG CAPTAIN3_PATH=/usr/share/captain3
ARG VENV_PATH=${CAPTAIN3_PATH}/venv
ARG EXAMPLES_PATH=${CAPTAIN3_PATH}/examples
ARG TMP_ZIP=/tmp/captain3_tmp.zip
ARG TRAINING_RESULTS_PATH=./output/training_results
ARG INFERENCE_RESULTS_PATH=./output/inference_results
ARG PLOTS_PATH=./output/plots

ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=Europe/Zurich

# Upgrade the packages in the container, install a few dependencies and fix the time zone
RUN apt-get update && apt-get upgrade -y && \
	apt-get install -y --no-install-recommends tzdata && \
	ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && \
	echo ${TZ} > /etc/timezone && \
	dpkg-reconfigure -f noninteractive tzdata && \
	apt-get install -y gdal-bin libgdal-dev && \
	rm -rf /var/lib/apt/lists/*

# Install `uv` directly in `/usr/bin`
RUN curl -LsSf https://astral.sh/uv/install.sh | env -i XDG_BIN_HOME=/usr/bin sh --noprofile --norc -l -s -- -q --no-modify-path

# PyTorch is alltogether removed from CAPTAIN dependencies in order to force the usage of the container PyTorch version (which is the only one compatible with the Alps ecosystem)
# CAPTAIN v.3.0 is patched in order to be retro-compatible with Python 3.10
RUN --mount=type=bind,source=./constraints.txt,target=/tmp/constraints.txt \
	--mount=type=bind,source=./captain3preview.patch,target=/tmp/captain3preview.patch \
	git clone https://github.com/captain-project/captain3preview /tmp/captain3preview && \
	patch -f -p1 -d /tmp/captain3preview < /tmp/captain3preview.patch && \
	sed -i -e 's/numpy[><=.,0-9]*/numpy>=1.23/' -e 's/\(requires-python[[:blank:]]*=[[:blank:]]*"\).*\("\)/\1>=3.10\2/' \
	-e '/^[[:blank:]]*"torch[<>=,.0-9]*",*[[:blank:]]*$/d' \
	-e '/\[\[tool\.uv\.index\]\]/,/^$/d' \
	-e '/\[tool\.uv\.sources\]/,/^$/d' \
        -e '$a\\n[tool.uv]\noverride-dependencies = [\n    "torch; sys_platform == '"'"'never'"'"'",\n]' /tmp/captain3preview/pyproject.toml && \
	uv venv --python "$(which python3)" --system-site-packages --seed --relocatable --link-mode=copy "${VENV_PATH}" && \
	uv pip install --python "${VENV_PATH}/bin/python" -c /tmp/constraints.txt setuptools hatchling && \
	uv pip install --python "${VENV_PATH}/bin/python" --no-build-isolation -c /tmp/constraints.txt /tmp/captain3preview && \
	rm -rf /tmp/captain3preview

# The examples python scripts, data and pre-trained model are downloaded into the container; the paths on the example python scripts are adjusted
RUN mkdir -p ${EXAMPLES_PATH} && \
	curl -LsSf -o "${EXAMPLES_PATH}/plot_input_data.py" https://github.com/captain-project/captain3preview/raw/refs/heads/main/examples/plot_input_data.py && \
	curl -LsSf -o "${EXAMPLES_PATH}/train_policy.py" https://github.com/captain-project/captain3preview/raw/refs/heads/main/examples/train_policy.py && \
	curl -LsSf -o "${EXAMPLES_PATH}/run_inference.py" https://github.com/captain-project/captain3preview/raw/refs/heads/main/examples/run_inference.py && \
	curl -LsSf -o "${TMP_ZIP}" https://polybox.ethz.ch/index.php/s/WKdbHHGj3ayL9w9/download/captain3data.zip && unzip -d "${CAPTAIN3_PATH}" "${TMP_ZIP}" && rm -f "${TMP_ZIP}" && \
	curl -LsSf -o "${TMP_ZIP}" https://polybox.ethz.ch/index.php/s/wZ5AMXPdzboZSm2/download/captain3trained_model.zip && unzip -d "${CAPTAIN3_PATH}" "${TMP_ZIP}" && rm -f "${TMP_ZIP}" && \
	sed -i -e 's|\([[:blank:]]*DATA_DIR[[:blank:]]*=[[:blank:]]*Path("\).*\(")\)|\1'"${CAPTAIN3_PATH}/captain3data"'\2|' \
	       -e 's|DATA_DIR / results_dir|Path("'"${PLOTS_PATH}"'")|' "${EXAMPLES_PATH}/plot_input_data.py" && \
	sed -i -e 's|\([[:blank:]]*DATA_DIR[[:blank:]]*=[[:blank:]]*Path("\).*\(")\)|\1'"${CAPTAIN3_PATH}/captain3data"'\2|' \
	       -e 's|\([[:blank:]]*RESULTS_DIR[[:blank:]]*=[[:blank:]]*\).*[[:blank:]]*|\1Path("'"${TRAINING_RESULTS_PATH}"'")|' "${EXAMPLES_PATH}/train_policy.py" && \
	sed -i -e 's|\([[:blank:]]*DATA_DIR[[:blank:]]*=[[:blank:]]*Path("\).*\(")\)|\1'"${CAPTAIN3_PATH}/captain3data"'\2|' \
	       -e 's|\([[:blank:]]*TRAINED_MODEL[[:blank:]]*=[[:blank:]]*Path("\).*\(")\)|\1'"${CAPTAIN3_PATH}/captain3trained_model/trained_weights.npy"'\2|' \
	       -e 's|DATA_DIR / results_dir|Path("'"${INFERENCE_RESULTS_PATH}"'")|' "${EXAMPLES_PATH}/run_inference.py"
