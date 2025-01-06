# Use uma imagem base do Python
FROM python:3.11-slim

# Configura diretório de trabalho no container
WORKDIR /app

# Instala dependências
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Cria um usuário não-root
RUN useradd -m appuser

# Copia o código do host para o container
COPY . /app

# Ajusta permissões para o novo usuário
RUN chown -R appuser /app

# Alterna para o usuário não-root
USER appuser

# # Instala pipreqs e gera requirements.txt
# RUN pip install --user pipreqs && #     ~/.local/bin/pipreqs /app --force --scan-notebooks --mode 'gt'

# # Instala dependências
RUN pip install --user --no-cache-dir -r requirements.txt

# Install the LoadDataset package from GitHub and the requirements
RUN pip install git+https://github.com/Olavo-B/LoadDataset

# Comando padrão
CMD ["bash"]
