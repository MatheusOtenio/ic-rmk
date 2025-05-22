# Pacote necessário
library(ggplot2)

# Dados atualizados
curso = c("Bacharelado Em Zootecnia", 
          "Bacharelado Em Agronomia", 
          "Engenharia Florestal", 
          "Licenciatura Em Ciências Biológicas", 
          "Bacharelado Em Engenharia De Software", 
          "Engenharia De Bioprocessos E Biotecnologia", 
          "Licenciatura Em Educação No Campo")
value = c(1224, 1156, 907, 902, 885, 432, 150)
df = data.frame(curso, value)

# Ordenando os dados
df = df[order(df$value, decreasing = FALSE),]
df$curso = factor(df$curso, levels = df$curso)

# Gráfico com nova paleta de cores e ajuste no limite do eixo y
g = ggplot(df, aes(x = curso, y = value, fill = curso)) 
g = g + geom_bar(stat = "identity") + theme_bw() 
g = g + geom_text(aes(label = value), colour = "black", hjust = 0.7)  # Ajuste para exibir fora da barra
g = g + labs(x = "Curso", y = "Número de Registros") + coord_flip()
g = g + scale_fill_manual(values = c("#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494"))
g = g + guides(fill = "none")
g = g + ylim(0, max(df$value) * 1.1)  # Limite aumentado para dar mais espaço

# Salvando o gráfico como PDF
ggsave(g, file = "course_counts_last_data.pdf", width = 5.73, height = 2.83)
