"""
    FILE MAIN.PY 

    AUTORI: 
    - Biancini Mattia 865966
    - Gargiulo Elio 869184

    DESCRIZIONE:
    File ui del progetto. 
    Contiene le funzioni necessarie per la creazione della ui con la
    liberia tkinter, il passaggio dei parametri necessari ed esecuzione della
    compressione dell'immagine bitmap.

"""
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
from functions import validate_data, compress_image

# Aumento max limite per la dimensione delle immagini
Image.MAX_IMAGE_PIXELS = None
selected_image = None
processed_image = None

# Selezione dell'immagine Bitmap
def browse_image(img_label, enable_process_button, size_label, proc_img_label):
    global selected_image, processed_image
    filename = filedialog.askopenfilename(filetypes=[("Bitmap files", "*.bmp")])
    if filename:
        try:
            selected_image = Image.open(filename).convert('L')
        except Image.DecompressionBombError:
            messagebox.showwarning("Attenzione", "L'immagine è rilevata come ZipBomb.")
        img_label.config(text="Immagine selezionata: " + filename)
        enable_process_button(F_entry, d_entry, process_button)
        update_image_info(size_label)
        resize_and_display_image(img_label)
        # Rimuovi l'immagine elaborata dalla UI se presente
        proc_img_label.config(image='')
        processed_image = None
        save_button.config(state=tk.DISABLED)

# Abilita il pulsante per l'elaborazione dell'immagine se tutti i campi sono compilati
def enable_process_button(F_entry, d_entry, process_button):
    if all((F_entry.get(), d_entry.get(), selected_image)):
        process_button.config(state=tk.NORMAL)
    else:
        process_button.config(state=tk.DISABLED)

# Aggiorna le informazioni sull'immagine selezionata
def update_image_info(size_label):
    global selected_image
    if selected_image:
        width, height = selected_image.size
        size_label.config(text=f"Dimensione immagine: {width}x{height} pixel")

# Processa l'immagine con i parametri forniti
def process_image(F_entry, d_entry, img_label, proc_img_label):
    global processed_image
    try:
        F = int(F_entry.get())
        d = int(d_entry.get())
        # Valida i dati di input
        is_valid = validate_data(d, F)
        if not is_valid:
            messagebox.showerror("ERRORE", "d non è compreso tra 0 e 2*F - 2")
            return
        # Crea un'immagine nera se d è 0, altrimenti comprimi l'immagine
        if d == 0:
            width, height = selected_image.size
            # Utilizzo le dimensioni considerando il taglio dei blocchi
            processed_image = Image.new('L', (F * (width // F), F * (height // F)), (0))
        else:
            processed_image = compress_image(selected_image, F, d)
        # Ridimensiona l'immagine elaborata per la visualizzazione
        resize_processed_image = processed_image.resize(get_resized_dimensions(processed_image))
        processed_image_tk = ImageTk.PhotoImage(resize_processed_image)
        proc_img_label.config(image=processed_image_tk)
        proc_img_label.image = processed_image_tk
        # Abilita il pulsante per salvare l'immagine
        save_button.config(state=tk.NORMAL)
    except Exception as e:
        messagebox.showerror("ERRORE", f"Impossibile Elaborare l'immagine: {str(e)}")

# Calcola le nuove dimensioni dell'immagine mantenendo il rapporto di aspetto
def get_resized_dimensions(image):
    window_height = root.winfo_height()
    window_width = root.winfo_width()
    max_height = window_height * 1.9 // 3  
    max_width = window_width // 2.1 
    original_width, original_height = image.size
    ratio = min(max_height / original_height, max_width / original_width)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    return (new_width, new_height)

# Ridimensiona e mostra l'immagine selezionata
def resize_and_display_image(img_label):
    if selected_image:
        resized_image = selected_image.resize(get_resized_dimensions(selected_image), Image.LANCZOS)
        image_tk = ImageTk.PhotoImage(resized_image)
        img_label.config(image=image_tk)
        img_label.image = image_tk

# Salva l'immagine elaborata
def save_image():
    if processed_image:
        save_filename = filedialog.asksaveasfilename(defaultextension=".bmp", filetypes=[("Bitmap files", "*.bmp")])
        if save_filename:
            processed_image.save(save_filename, "BMP")
            messagebox.showinfo("Successo", "Immagine salvata con successo.")

# Creazione della finestra principale
root = tk.Tk()
root.title("Compressione Immagini - Progetto 2")

# Dimensioni fisse della finestra
window_width = 1000
window_height = 700

# Posizionamento della finestra al centro dello schermo
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_position = (screen_width - window_width) // 2
y_position = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
root.resizable(False, False)
root.configure(padx=10, pady=10)

# Frame per contenere le immagini
frame = tk.Frame(root)
frame.pack()

# Etichetta per visualizzare l'immagine originale
img_label = tk.Label(frame)
img_label.grid(row=0, column=0, padx=5, pady=5)

# Etichetta per visualizzare l'immagine processata
proc_img_label = tk.Label(frame)
proc_img_label.grid(row=0, column=1, padx=5, pady=5)

# Frame per la parte inferiore con i controlli
bottom_frame = tk.Frame(root)
bottom_frame.pack(side=tk.BOTTOM, pady=10, fill=tk.X)

# Configurazione delle colonne per il layout centrato
bottom_frame.columnconfigure(0, weight=1)
bottom_frame.columnconfigure(1, weight=1)

# Pulsante per selezionare un'immagine
browse_button = ttk.Button(bottom_frame, text="Scegli Immagine", command=lambda: browse_image(img_label, enable_process_button, size_label, proc_img_label))
browse_button.grid(row=0, column=0, padx=10, pady=5, columnspan=2, sticky='ew')

# Etichetta per visualizzare la dimensione dell'immagine selezionata
size_label = tk.Label(bottom_frame, text="")
size_label.grid(row=1, column=0, padx=10, pady=5, columnspan=2, sticky='ew')

# Etichetta e campo di inserimento per l'ampiezza delle finestrelle (F)
F_label = tk.Label(bottom_frame, text="Ampiezza Finestrelle (F):")
F_label.grid(row=2, column=0, padx=10, pady=5, sticky='e')
F_entry = ttk.Spinbox(bottom_frame, from_=0, to=100, increment=1)
F_entry.grid(row=2, column=1, padx=10, pady=5, sticky='w')

# Etichetta e campo di inserimento per la soglia di taglio delle frequenze (d)
d_label = tk.Label(bottom_frame, text="Soglia di Taglio Frequenze (d):")
d_label.grid(row=3, column=0, padx=10, pady=5, sticky='e')
d_entry = ttk.Spinbox(bottom_frame, from_=0, to=100, increment=1)
d_entry.grid(row=3, column=1, padx=10, pady=5, sticky='w')

# Frame per contenere i pulsanti affiancati
button_frame = tk.Frame(bottom_frame)
button_frame.grid(row=4, column=0, columnspan=2, pady=5, sticky='ew')

# Configura le colonne del button_frame per estendersi
button_frame.columnconfigure(0, weight=1)
button_frame.columnconfigure(1, weight=1)

# Pulsante per elaborare l'immagine con i parametri inseriti
process_button = ttk.Button(button_frame, text="Elabora Immagine", command=lambda: process_image(F_entry, d_entry, img_label, proc_img_label), state=tk.DISABLED)
process_button.grid(row=0, column=0, padx=10, pady=5, sticky='ew')

# Pulsante per salvare l'immagine compressa
save_button = ttk.Button(button_frame, text="Salva Immagine Compressa", command=save_image, state=tk.DISABLED)
save_button.grid(row=0, column=1, padx=10, pady=5, sticky='ew')

# Abilito pulsante di elaborazione img
F_entry.bind("<KeyRelease>", lambda event: enable_process_button(F_entry, d_entry, process_button))
d_entry.bind("<KeyRelease>", lambda event: enable_process_button(F_entry, d_entry, process_button))

# Resize dell'immmagine (binding)
root.bind("<Configure>", lambda event: resize_and_display_image(img_label))

# Avvio del loop principale della finestra
root.mainloop()
