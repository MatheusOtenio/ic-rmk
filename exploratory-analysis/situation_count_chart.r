library(ggplot2)

status = c("Desistente", "Formado", "Regular", "Trancado", "Transferido", "Outros")
value = c(2376, 1638, 1337, 157, 118, 30)
df2 = data.frame(status, value)

df2 = df2[order(df2$value, decreasing=FALSE),]
df2$status = factor(df2$status, levels = df2$status)

g = ggplot(df2, mapping = aes(x = status, y = value, colour = status, fill = status)) 
g = g + geom_bar(stat = "identity") + theme_bw() 
g = g + geom_text(aes(label=value), colour = "black", hjust=0.6)
g = g + labs(x = "Situação", y = "Número de Registros") + coord_flip()
g = g + guides(fill = "none", colour = "none")
g
ggsave(g, file = "situation_counts_last_data.pdf",width=4.73,height=2.83)